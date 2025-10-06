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
import time
import random

import fsspec
import ray
from huggingface_hub import HfFileSystem
from tqdm_loggable.tqdm_logging import tqdm_logging

from marin.core.runtime import simple_backpressure
from marin.download.huggingface.download import DownloadConfig
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger("ray")


def ensure_fsspec_path_writable(output_path: str) -> None:
    fs, path = fsspec.core.url_to_fs(output_path)
    try:
        test_path = os.path.join(output_path, "test_write_access")
        with fs.open(test_path, "w") as f:
            f.write("test")
        fs.rm(test_path)
    except Exception as e:
        raise ValueError(f"No write access to fsspec path: {output_path} ({e})") from e


@ray.remote
def stream_file_to_fsspec(cfg: DownloadConfig, hf_fs: HfFileSystem, file_path: str, fsspec_file_path: str):
    target_fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)
    chunk_size = 16 * 1024 * 1024  # 16MB chunks for large files
    max_retries = 10

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
            logger.warning(f"Attempt {attempt+1} failed for {file_path}: {e}, retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed to download {file_path} after {max_retries} attempts")


def download_hf(cfg: DownloadConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)
    ensure_fsspec_path_writable(cfg.gcs_output_path)

    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", False))
    hf_repo_name_with_prefix = os.path.join(cfg.hf_repo_type_prefix, cfg.hf_dataset_id)

    files = []
    if not cfg.hf_urls_glob:
        files = hf_fs.find(hf_repo_name_with_prefix, revision=cfg.revision)
    else:
        for hf_url_glob in cfg.hf_urls_glob:
            pattern = os.path.join(hf_repo_name_with_prefix, hf_url_glob)
            files += hf_fs.glob(pattern, revision=cfg.revision)

    if not files:
        raise ValueError(f"No files found for dataset `{cfg.hf_dataset_id}`")

    task_generator = [(cfg, hf_fs, f, os.path.join(cfg.gcs_output_path, f.split("/", 3)[-1])) for f in files]

    logger.info(f"Total files to process: {len(task_generator)}")
    pbar = tqdm_logging(total=len(task_generator))

    max_in_flight = 8
    for ref in simple_backpressure(
        stream_file_to_fsspec, iter(task_generator), max_in_flight=max_in_flight, fetch_local=True
    ):
        ray.get(ref)
        pbar.update(1)

    write_provenance_json(
        cfg.gcs_output_path, metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": files}
    )
    logger.info(f"All files streamed successfully. Provenance JSON written to {cfg.gcs_output_path}.")
