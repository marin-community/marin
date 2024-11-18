#!/usr/bin/env python3
"""
A script to download a HuggingFace dataset and upload it to a specified fsspec path,
using HfFileSystem for direct streaming of data transfer.
"""

import logging
import os

import draccus
import fsspec
import ray
from huggingface_hub import HfFileSystem
from tqdm_loggable.auto import tqdm

from marin.core.runtime import simple_backpressure
from marin.utilities.validation_utils import write_provenance_json
from operations.download.huggingface.download import DownloadConfig

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


@ray.remote
def stream_file_to_fsspec(cfg: DownloadConfig, hf_fs: HfFileSystem, file_path: str, fsspec_file_path: str):
    """Ray task to stream a file from HfFileSystem to another fsspec path."""
    target_fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)

    try:
        with hf_fs.open(file_path, "rb") as src_file:
            with target_fs.open(fsspec_file_path, "wb") as dest_file:
                while chunk := src_file.read(1024 * 1024):  # Read in 1MB chunks
                    dest_file.write(chunk)
        logging.info(f"Streamed {file_path} to fsspec path: {fsspec_file_path}")
    except Exception as e:
        logging.exception(f"Error processing {file_path}: {e}")
        raise


def download_hf(cfg: DownloadConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Parse the output path and get the file system
    fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)

    # Use revision as "version" for writing to the output path
    versioned_output_path = os.path.join(cfg.gcs_output_path, cfg.revision)

    # Ensure the output path is writable
    try:
        ensure_fsspec_path_writable(versioned_output_path)
    except ValueError as e:
        logger.exception(f"Output path validation failed: {e}")
        raise e

    # Initialize Hugging Face filesystem
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN", False))

    # Get list of files directly from HfFileSystem matching the pattern
    pattern = f"datasets/{cfg.hf_dataset_id}/{cfg.hf_url_glob}"
    files = hf_fs.glob(pattern, revision=cfg.revision)

    task_generator = []

    for file in files:
        try:
            fsspec_file_path = os.path.join(versioned_output_path, file.split("/", 3)[-1])  # Strip the dataset prefix
            # Hf file paths are always of format : hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
            task_generator.append((cfg, hf_fs, file, fsspec_file_path))
        except Exception as e:
            logging.exception(f"Error preparing task for {file}: {e}")

    total_files = len(task_generator)
    logger.info(f"Total number of files to process: {total_files}")
    pbar = tqdm(total=total_files)

    for ref in simple_backpressure(stream_file_to_fsspec, iter(task_generator), max_in_flight=32, fetch_local=True):
        try:
            ray.get(ref)
            pbar.update(1)
        except Exception as e:
            logging.exception(f"Error during task execution: {e}")

    # Write Provenance JSON
    write_provenance_json(
        versioned_output_path,
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": files},
    )

    logger.info(f"Streamed all files and wrote provenance JSON; check {cfg.gcs_output_path}.")
