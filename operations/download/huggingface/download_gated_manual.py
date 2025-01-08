#!/usr/bin/env python3
"""
A script to download a gated HuggingFace dataset and upload it to a specified fsspec path,
preserving directory structures and handling different file types.

Run with (after setting HF_TOKEN as an environment variable):
    - [Local] python operations/download/huggingface/download_gated_manual.py \
          --hf_dataset_id EleutherAI/proof-pile-2 --revision main \
          --output_path fsspec://mybucket/raw/proof-pile-manual
"""

import fnmatch
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import draccus
import fsspec
from huggingface_hub import HfApi, hf_hub_download
from tqdm_loggable.auto import tqdm

from marin.utilities.validation_utils import write_provenance_json
from operations.download.huggingface.download import DownloadConfig

# Set up logging
logger = logging.getLogger(__name__)


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


def download_and_upload_to_store(cfg: DownloadConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    logging.warning(
        "DEPRECATED: This function is deprecated and will be removed in a future release." "Consider using download_hf"
    )

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
    hf_token = os.environ.get("HF_TOKEN")
    hf_client = HfApi(token=hf_token)

    # Get list of files in the dataset
    files = hf_client.list_repo_files(repo_id=cfg.hf_dataset_id, revision=cfg.revision, repo_type="dataset")

    total_files = len(files)
    logger.info(f"Total number of files to process: {total_files}")
    pbar = tqdm(total=total_files)

    thread_pool = ThreadPoolExecutor(max_workers=32)

    def put_file(temp_dir, hf_url, fsspec_file_path):
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
            logging.info(f"Uploaded {file} to fsspec path: {fsspec_file_path}")

            os.remove(local_path)
        except Exception as e:
            logging.exception(f"Error processing {file}: {e}")
            # we want to fail the whole operation if any file fails to upload
            raise

        pbar.update(1)

    hf_urls = []
    futures = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            for hf_url_glob in cfg.hf_urls_glob:
                if fnmatch.fnmatch(file, hf_url_glob):
                    try:
                        # Construct HuggingFace URL
                        hf_url = construct_hf_url(cfg.hf_dataset_id, cfg.revision, file)
                        hf_urls.append(hf_url)
                        fsspec_file_path = os.path.join(versioned_output_path, file)

                        f = thread_pool.submit(put_file, temp_dir, file, fsspec_file_path)
                        futures.append((file, f))

                    except Exception as e:
                        logging.exception(f"Error processing {file}: {e}")

        for file, f in futures:
            try:
                f.result()
            except Exception as e:
                logging.exception(f"Error processing {file}: {e}")

    # Write Provenance JSON
    write_provenance_json(
        Path(versioned_output_path),
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": hf_urls},
    )

    logger.info(f"Uploaded all files and wrote provenance JSON; check {cfg.gcs_output_path}.")


@draccus.wrap()
def download_gated_main(cfg: DownloadConfig):
    download_and_upload_to_store(cfg)


if __name__ == "__main__":
    download_gated_main()
