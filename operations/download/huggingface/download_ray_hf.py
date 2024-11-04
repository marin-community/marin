#!/usr/bin/env python3
"""
A script to download a gated HuggingFace dataset and upload it to a specified fsspec path,
preserving directory structures and handling different file types.
"""

import fnmatch
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import draccus
import fsspec
import ray
from huggingface_hub import HfApi, hf_hub_download
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


def construct_hf_url(dataset_id: str, revision: str, file_path: str) -> str:
    """Construct a Hugging Face dataset URL manually."""
    encoded_file_path = quote(file_path)
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{encoded_file_path}"


@ray.remote
def put_file(cfg: DownloadConfig, temp_dir: os.PathLike, hf_url: str, fsspec_file_path: os.PathLike):
    """Ray task to download repo onto local disk before uploading it onto remote."""
    fs, _ = fsspec.core.url_to_fs(cfg.gcs_output_path)
    hf_token = os.environ.get("HF_TOKEN")

    try:
        # Download file from HuggingFace
        local_path = hf_hub_download(
            repo_id=cfg.hf_dataset_id,
            filename=hf_url,
            revision=cfg.revision,
            token=hf_token,
            local_dir=temp_dir,
            repo_type="dataset",
        )

        # Upload file using fsspec
        fs.put(local_path, fsspec_file_path)
        logging.info(f"Uploaded {hf_url} to fsspec path: {fsspec_file_path}")

        os.remove(local_path)
    except Exception as e:
        logging.exception(f"Error processing {hf_url}: {e}")
        raise


@draccus.wrap()
def download_ray_hf(cfg: DownloadConfig) -> None:
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

    # Initialize HuggingFace client
    hf_token = os.environ.get("HF_TOKEN", False)
    hf_client = HfApi(token=hf_token)

    # Get list of files in the dataset
    files = hf_client.list_repo_files(repo_id=cfg.hf_dataset_id, revision=cfg.revision, repo_type="dataset")

    hf_urls = []
    task_generator = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            if fnmatch.fnmatch(file, cfg.hf_url_glob):
                try:
                    # Construct HuggingFace URL
                    hf_url = construct_hf_url(cfg.hf_dataset_id, cfg.revision, file)
                    hf_urls.append(hf_url)
                    fsspec_file_path = os.path.join(versioned_output_path, file)
                    task_generator.append((cfg, temp_dir, file, fsspec_file_path))
                except Exception as e:
                    logging.exception(f"Error preparing task for {file}: {e}")

        total_files = len(hf_urls)
        logger.info(f"Total number of files to process: {total_files}")
        pbar = tqdm(total=total_files)

        for ref in simple_backpressure(put_file, iter(task_generator), max_in_flight=32, fetch_local=True):
            try:
                ray.get(ref)
                pbar.update(1)
            except Exception as e:
                logging.exception(f"Error during task execution: {e}")

    # Write Provenance JSON
    write_provenance_json(
        Path(versioned_output_path),
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": hf_urls},
    )

    logger.info(f"Uploaded all files and wrote provenance JSON; check {cfg.gcs_output_path}.")
