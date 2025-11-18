#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A script to download a HuggingFace dataset and upload it to a specified fsspec path,
using HfFileSystem for direct streaming of data transfer.
"""

import dataclasses
import logging
import os
import random
import time
from dataclasses import dataclass

import draccus
import fsspec
from huggingface_hub import HfFileSystem
from marin.execution.executor import THIS_OUTPUT_PATH
from marin.utilities.validation_utils import write_provenance_json
from zephyr import Dataset, flow_backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    # fmt: off

    # HuggingFace Dataset Parameters
    hf_dataset_id: str                                      # HF Dataset to Download (as `$ORG/$DATASET` on HF Hub)

    revision: str  # (Short) Commit Hash (from HF Dataset Repo; 7 characters)
    hf_urls_glob: list[str] = dataclasses.field(default_factory=list)
    # List of Glob Patterns to Match Files in HF Dataset, If empty we get all the files in a hf repo

    gcs_output_path: str = THIS_OUTPUT_PATH
    """
    Path to store raw data in persistent storage (e.g. gs://$BUCKET/...).
    This works with any fsspec-compatible path, but for backwards compatibility, we call it gcs_output_path.
    """

    append_sha_to_path: bool = False
    """If true, write outputs under ``gcs_output_path/<revision>`` instead of directly under ``gcs_output_path``."""

    # Job Control Parameters, used only for non-gated dataset transfers done via STS
    wait_for_completion: bool = True                        # if True, will block until job completes

    # fmt: on
    hf_repo_type_prefix: str = (
        "datasets"  # The repo_type_prefix is datasets/ for datasets,
        # spaces/ for spaces, and models do not need a prefix in the URL.
    )


def ensure_fsspec_path_writable(output_path: str) -> None:
    """Check if the fsspec path is writable by trying to create and delete a temporary file."""
    fs, _ = fsspec.core.url_to_fs(output_path)
    try:
        fs.mkdirs(output_path, exist_ok=True)
        test_path = os.path.join(output_path, "test_write_access")
        with fs.open(test_path, "w") as f:
            f.write("test")
        fs.rm(test_path)
    except Exception as e:
        raise ValueError(f"No write access to fsspec path: {output_path} ({e})") from e


def stream_file_to_fsspec(gcs_output_path: str, file_path: str, fsspec_file_path: str):
    """Ray task to stream a file from HfFileSystem to another fsspec path."""
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", False))
    target_fs, _ = fsspec.core.url_to_fs(gcs_output_path)
    # Use larger chunk size for large files, such as 32B models
    chunk_size = 16 * 1024 * 1024 * 1024
    max_retries = 10

    # Retry when there is an error, such as hf rate limit
    for attempt in range(max_retries):
        try:
            with hf_fs.open(file_path, "rb") as src_file:
                target_fs.mkdirs(os.path.dirname(fsspec_file_path), exist_ok=True)
                with target_fs.open(fsspec_file_path, "wb") as dest_file:
                    while chunk := src_file.read(chunk_size):
                        dest_file.write(chunk)
            logger.info(f"Streamed {file_path} successfully to {fsspec_file_path}")
            return
        except Exception as e:
            wait_time = (2**attempt) + random.uniform(0, 5)
            logger.warning(f"Attempt {attempt + 1} failed for {file_path}: {e}, retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed to download {file_path} after {max_retries} attempts")


def download_hf(cfg: DownloadConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Set cfg.append_sha_to_path=True to mimic the older behavior of writing to gcs_output_path/<revision>.
    # Some historical datasets were written that way, so this flag keeps backwards compatibility when needed.

    # Ensure the output path is writable
    try:
        output_path = os.path.join(cfg.gcs_output_path, cfg.revision) if cfg.append_sha_to_path else cfg.gcs_output_path
        ensure_fsspec_path_writable(output_path)
    except ValueError as e:
        logger.exception(f"Output path validation failed: {e}")
        raise e

    # Initialize Hugging Face filesystem
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", False))
    hf_repo_name_with_prefix = os.path.join(cfg.hf_repo_type_prefix, cfg.hf_dataset_id)

    if not cfg.hf_urls_glob:
        # We get all the files using find
        files = hf_fs.find(hf_repo_name_with_prefix, revision=cfg.revision)
    else:
        # Get list of files directly from HfFileSystem matching the pattern
        files = []
        for hf_url_glob in cfg.hf_urls_glob:
            pattern = os.path.join(hf_repo_name_with_prefix, hf_url_glob)
            files += hf_fs.glob(pattern, revision=cfg.revision)

    if not files:
        raise ValueError(f"No files found for dataset `{cfg.hf_dataset_id}. Used glob patterns: {cfg.hf_urls_glob}")

    task_generator = []

    for file in files:
        try:
            fsspec_file_path = os.path.join(output_path, file.split("/", 3)[-1])  # Strip the dataset prefix
            # Hf file paths are always of format : hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
            task_generator.append((output_path, file, fsspec_file_path))
        except Exception as e:
            logging.exception(f"Error preparing task for {file}: {e}")

    total_files = len(task_generator)
    logger.info(f"Total number of files to process: {total_files}")

    backend = flow_backend(max_parallelism=512)
    pipeline = (
        Dataset.from_list(task_generator)
        .map(lambda task: stream_file_to_fsspec(*task))
        .write_jsonl(
            f"{cfg.public_gcs_path}/.metrics/success-part-{{shard:05d}}-of-{{num_shards:05d}}.jsonl", skip_existing=True
        )
    )
    list(backend.execute(pipeline))

    # Write Provenance JSON
    write_provenance_json(
        output_path,
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": files},
    )

    logger.info(f"Streamed all files and wrote provenance JSON; check {output_path}.")


@draccus.wrap()
def main(cfg: DownloadConfig) -> None:
    """Download HuggingFace dataset."""
    download_hf(cfg)


if __name__ == "__main__":
    main()
