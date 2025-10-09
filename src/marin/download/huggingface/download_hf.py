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

import logging
import os

import fsspec
import ray
from huggingface_hub import HfFileSystem
from tqdm_loggable.tqdm_logging import tqdm_logging

from marin.core.runtime import simple_backpressure
from marin.download.huggingface.download import DownloadConfig
from marin.utilities.validation_utils import write_provenance_json

# Set up logging
logger = logging.getLogger("ray")


def ensure_fsspec_path_writable(output_path: str) -> None:
    """Check if the fsspec path is writable by trying to create and delete a temporary file."""
    fs, path = fsspec.core.url_to_fs(output_path)
    try:
        test_path = os.path.join(output_path, "test_write_access")
        with fs.open(test_path, "w") as f:
            f.write("test")
        fs.rm(test_path)
    except Exception as e:
        raise ValueError(f"No write access to fsspec path: {output_path} ({e})") from e


@ray.remote(max_retries=5, retry_exceptions=True)
def stream_file_to_fsspec(cfg: DownloadConfig, hf_fs: HfFileSystem, file_path: str, fsspec_file_path: str):
    """Ray task to stream a file from HfFileSystem to another fsspec path."""
    target_fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)

    # Tuning knobs (env-overridable)
    timeout = int(os.environ.get("HF_TIMEOUT_SEC", "600"))
    block_size_mb = int(os.environ.get("HF_BLOCK_SIZE_MB", "32"))
    block_size = block_size_mb * 1024 * 1024

    try:
        # Use larger block sizes to reduce HTTP range request count
        with hf_fs.open(file_path, "rb", timeout=timeout, block_size=block_size) as src_file:
            target_fs.mkdirs(os.path.dirname(fsspec_file_path), exist_ok=True)
            with target_fs.open(fsspec_file_path, "wb") as dest_file:
                # Stream in sizable chunks (independent from fsspec block_size)
                chunk_size_mb = int(os.environ.get("HF_CHUNK_SIZE_MB", str(max(8, block_size_mb))))
                chunk_size = chunk_size_mb * 1024 * 1024
                while chunk := src_file.read(chunk_size):
                    dest_file.write(chunk)
        logging.info(f"Streamed {file_path} to fsspec path: {fsspec_file_path}")
    except Exception as e:
        logging.exception(f"Error processing {file_path}: {e}")
        raise


def download_hf(cfg: DownloadConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Parse the output path and get the file system
    fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)

    # TODO: Our earlier version of download_hf used this piece of code for calculating the versioned_output_path
    # versioned_output_path = os.path.join(cfg.gcs_output_path, cfg.revision)
    # This versioned_output_path was used instead of gcs_output_path. So some of the earlier datasets are stored in
    # gcs_output_path/<revision> instead of gcs_output_path. We should do this migration.

    # Ensure the output path is writable
    try:
        ensure_fsspec_path_writable(cfg.gcs_output_path)
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
            fsspec_file_path = os.path.join(cfg.gcs_output_path, file.split("/", 3)[-1])  # Strip the dataset prefix
            # Hf file paths are always of format : hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
            task_generator.append((cfg, hf_fs, file, fsspec_file_path))
        except Exception as e:
            logging.exception(f"Error preparing task for {file}: {e}")

    total_files = len(task_generator)
    logger.info(f"Total number of files to process: {total_files}")
    pbar = tqdm_logging(total=total_files)

    # Limit per-dataset inflight to avoid HF 429; overridable via env
    max_in_flight = int(os.environ.get("MARIN_HF_MAX_IN_FLIGHT", "8"))

    for ref in simple_backpressure(
        stream_file_to_fsspec,
        iter(task_generator),
        max_in_flight=max_in_flight,
        fetch_local=True,
    ):
        try:
            ray.get(ref)
            pbar.update(1)
        except Exception as e:
            logging.exception(f"Error during task execution: {e}")
            raise e

    # Write Provenance JSON
    write_provenance_json(
        cfg.gcs_output_path,
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": files},
    )

    logger.info(f"Streamed all files and wrote provenance JSON; check {cfg.gcs_output_path}.")
