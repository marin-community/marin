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
import random
import time
from dataclasses import dataclass, field

import draccus
import fsspec
from huggingface_hub import HfFileSystem
from huggingface_hub.errors import HfHubHTTPError
from marin.execution.executor import THIS_OUTPUT_PATH
from marin.utilities.validation_utils import write_provenance_json
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    # fmt: off

    # HuggingFace Dataset Parameters
    hf_dataset_id: str                                      # HF Dataset to Download (as `$ORG/$DATASET` on HF Hub)

    revision: str  # (Short) Commit Hash (from HF Dataset Repo; 7 characters)
    hf_urls_glob: list[str] = field(default_factory=list)
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

    zephyr_max_parallelism: int = 32
    """Maximum parallelism of the Zephyr download job"""


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


def stream_file_to_fsspec(gcs_output_path: str, file_path: str, fsspec_file_path: str, expected_size: int | None = None):
    """Stream a file from HfFileSystem to another fsspec path using atomic write.

    Uses atomic_rename to write to a temp file first, then rename on success.
    This enables recovery across individual files if the job is interrupted.

    Args:
        gcs_output_path: Base output path for the download.
        file_path: Source file path on HuggingFace.
        fsspec_file_path: Target file path on the destination filesystem.
        expected_size: Expected file size in bytes for validation. If provided,
            the download will fail if the downloaded size doesn't match.
    """
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", False))
    target_fs, _ = fsspec.core.url_to_fs(gcs_output_path)
    # Use 256 MB chunk size for large files
    chunk_size = 256 * 1024 * 1024
    max_retries = 20
    # 15 minutes max sleep
    max_sleep = 15 * 60
    # Minimum base wait time to avoid too-fast retries
    min_base_wait = 5

    # Retry when there is an error, such as hf rate limit
    last_exception = None
    for attempt in range(max_retries):
        try:
            target_fs.mkdirs(os.path.dirname(fsspec_file_path), exist_ok=True)
            bytes_written = 0
            with atomic_rename(fsspec_file_path) as temp_path:
                with hf_fs.open(file_path, "rb") as src_file, fsspec.open(temp_path, "wb") as dest_file:
                    while chunk := src_file.read(chunk_size):
                        dest_file.write(chunk)
                        bytes_written += len(chunk)

                # Validate file size BEFORE atomic_rename commits the file
                if expected_size is not None and bytes_written != expected_size:
                    raise ValueError(
                        f"Size mismatch for {file_path}: expected {expected_size} bytes, got {bytes_written} bytes"
                    )

            logger.info(f"Streamed {file_path} successfully to {fsspec_file_path} ({bytes_written} bytes)")
            return {"file_path": file_path, "status": "success", "size": bytes_written}
        except Exception as e:
            last_exception = e
            # Base wait: min 5s, then exponential: 5, 10, 20, 40, 80, 160, 320, 600 (capped)
            wait_base = max(min_base_wait, min_base_wait * (2**attempt))

            error_type = type(e).__name__
            error_msg = str(e)
            status_code = -1

            if isinstance(e, HfHubHTTPError):
                status_code = e.response.status_code
                TOO_MANY_REQUESTS = 429
                if status_code == TOO_MANY_REQUESTS:
                    # NOTE: RateLimit "api\|pages\|resolvers";r=[remaining];t=[seconds remaining until reset]
                    try:
                        rate_limit_wait = int(e.response.headers["RateLimit"].split(";")[-1].split("=")[-1])
                        wait_base = max(wait_base, rate_limit_wait + 10)  # Add buffer to rate limit wait
                    except Exception:
                        logger.warning("Failed to parse rate limit header, using default wait period")

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {file_path}: "
                f"{error_type} (status={status_code}): {error_msg}"
            )

            jitter = random.uniform(0, min(wait_base * 0.25, 30))  # Up to 25% jitter, max 30s
            wait_time = min(wait_base + jitter, max_sleep)

            logger.info(f"Retrying {file_path} in {wait_time:.1f}s...")
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to download {file_path} after {max_retries} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    )


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
    logger.info("Identifying files to download from HuggingFace...")
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

    # Get file sizes for validation
    logger.info("Getting file sizes for validation...")
    file_sizes: dict[str, int | None] = {}
    for file in files:
        try:
            info = hf_fs.info(file, revision=cfg.revision)
            file_sizes[file] = info.get("size") or None
        except Exception as e:
            logger.warning(f"Could not get size for {file}: {e}")
            file_sizes[file] = None  # Will skip validation for this file

    download_tasks = []

    for file in files:
        try:
            fsspec_file_path = os.path.join(output_path, file.split("/", 3)[-1])  # Strip the dataset prefix
            # Hf file paths are always of format : hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
            expected_size = file_sizes.get(file)
            download_tasks.append((output_path, file, fsspec_file_path, expected_size))
        except Exception as e:
            logging.exception(f"Error preparing task for {file}: {e}")

    total_files = len(download_tasks)
    total_size_gb = sum(s for s in file_sizes.values() if s is not None) / (1024**3)
    logger.info(f"Total number of files to process: {total_files} ({total_size_gb:.2f} GB)")

    pipeline = (
        Dataset.from_list(download_tasks)
        .map(lambda task: stream_file_to_fsspec(*task))
        .write_jsonl(
            f"{cfg.gcs_output_path}/.metrics/success-part-{{shard:05d}}-of-{{total:05d}}.jsonl", skip_existing=True
        )
    )
    with ZephyrContext(name="download-hf") as ctx:
        ctx.execute(pipeline)

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
