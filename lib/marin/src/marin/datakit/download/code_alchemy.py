# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream Code Alchemy parquet shards into subset and ``blob_id`` partitions."""

import logging
from dataclasses import dataclass, field

import draccus
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from marin.datakit.download.blob_id import (
    BLOB_NULL_PARTITION,
    BLOB_PREFIX_HEX_WIDTH,
    BLOB_PREFIX_PARTITION_COUNT,
    BlobIdDownloadConfig,
    BlobIdParquetTask,
    execute_blob_id_partition_download,
    list_blob_id_parquet_tasks,
)

logger = logging.getLogger(__name__)

HF_DATASET_ID = "open-alchemy/code-alchemy"
HF_REVISION = "d367da91def5024929d0fa8d46d47d4ef616b467"
HF_PARQUET_GLOB = "*/*.parquet"


@dataclass(frozen=True)
class CodeAlchemyDownloadConfig(BlobIdDownloadConfig):
    """Configuration for the direct Hugging Face-to-object-store transfer."""

    output_path: str = field(default_factory=lambda: marin_temp_bucket(ttl_days=30, prefix="code-alchemy"))
    revision: str = HF_REVISION


def list_code_alchemy_tasks(cfg: CodeAlchemyDownloadConfig) -> list[BlobIdParquetTask]:
    """List pinned parquet files, preserving each subset's distinct schema."""

    data_output_path = prefix_join(cfg.output_path, "data")
    return list_blob_id_parquet_tasks(
        cfg,
        dataset_id=HF_DATASET_ID,
        parquet_glob=HF_PARQUET_GLOB,
        require_token=False,
        output_path_for_relative=lambda path: prefix_join(
            data_output_path,
            f"subset={path.partition('/')[0]}",
        ),
    )


def download_code_alchemy(cfg: CodeAlchemyDownloadConfig) -> None:
    """List the repository and execute the direct streaming transfer."""

    tasks = list_code_alchemy_tasks(cfg)
    subsets = sorted({task.relative_source_path.partition("/")[0] for task in tasks})
    logger.info(
        "Downloading %d parquet files across %d subsets with up to %d workers into "
        "%d hexadecimal blob-prefix partitions plus %s",
        len(tasks),
        len(subsets),
        min(cfg.max_workers, len(tasks)),
        BLOB_PREFIX_PARTITION_COUNT,
        BLOB_NULL_PARTITION,
    )
    execute_blob_id_partition_download(
        tasks,
        cfg,
        job_name="download-code-alchemy",
        provenance_metadata={
            "dataset": HF_DATASET_ID,
            "revision": cfg.revision,
            "source_glob": HF_PARQUET_GLOB,
            "source_file_count": len(tasks),
            "subsets": subsets,
            "partition_key": f"subset,lower(blob_id[:{BLOB_PREFIX_HEX_WIDTH}])",
            "blob_partition_count_per_subset": BLOB_PREFIX_PARTITION_COUNT + 1,
            "null_blob_partition": BLOB_NULL_PARTITION,
        },
    )


@draccus.wrap()
def main(cfg: CodeAlchemyDownloadConfig) -> None:
    configure_logging(level=logging.INFO)
    download_code_alchemy(cfg)


if __name__ == "__main__":
    main()
