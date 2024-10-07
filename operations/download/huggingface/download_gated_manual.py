#!/usr/bin/env python3
"""
improved_download_gated_hf_dataset.py

An improved script to download a gated HuggingFace dataset and upload it to a specified GCS path,
preserving directory structures and handling different file types.

Run with (after setting HF_TOKEN as an environment variable):
    - [Local] python operations/download/huggingface/download_gated_manual.py \
          --hf_dataset_id EleutherAI/proof-pile-2 --revision main \
          --gcs_output_path gs://marin-us-central2/raw/proof-pile-manual
"""

import os
import argparse
import logging
from pathlib import Path
import tempfile
import shutil
import fnmatch
from urllib.parse import quote
import draccus

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions
from marin.utilities.gcs_utils import split_gcs_path
from marin.utilities.validation_utils import write_provenance_json
from operations.download.huggingface.download import DownloadConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_gcs_path_exists(gcs_path):
    gcs_client = storage.Client()
    bucket_name, blob_prefix = gcs_path.replace("gs://", "").split("/", 1)
    bucket = gcs_client.bucket(bucket_name)

    if not bucket.exists():
        raise ValueError(f"GCS bucket {bucket_name} does not exist.")
    
    print(f"Checking write access to GCS path: gs://{bucket_name}/{blob_prefix}")

    # Check if we can write to the specified path
    test_blob = bucket.blob(f"{blob_prefix}/test_write_access")
    try:
        test_blob.upload_from_string("test")
        test_blob.delete()
    except gcp_exceptions.Forbidden:
        raise ValueError(f"No write access to GCS path: gs://{bucket_name}/{blob_prefix}")
    

def construct_hf_url(dataset_id: str, revision: str, file_path: str) -> str:
    """Construct a Hugging Face dataset URL manually."""
    encoded_file_path = quote(file_path)
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{encoded_file_path}"


def download_and_upload_to_gcs(cfg: DownloadConfig) -> None:

    # Parse GCS Bucket, Relative Path from `gcs_output_path`
    gcs_bucket, gcs_relative_path = split_gcs_path(cfg.gcs_output_path)

    # Use `revision` as "version" for writing to GCS
    gcs_versioned_relative_path = os.path.join(gcs_relative_path, cfg.revision)

    # Construct full GCS path
    full_gcs_path = f"gs://{gcs_bucket}/{gcs_versioned_relative_path}"

    # Ensure GCS path exists and is writable
    try:
        ensure_gcs_path_exists(full_gcs_path)
    except ValueError as e:
        logging.error(f"GCS path validation failed: {str(e)}")
        return

    # Initialize HuggingFace client
    hf_token = os.environ.get('HF_TOKEN')
    hf_client = HfApi(token=hf_token)

    # Initialize GCS client
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket)

    # Get list of files in the dataset
    files = hf_client.list_repo_files(repo_id=cfg.hf_dataset_id, revision=cfg.revision, repo_type="dataset")

    # construct HF urls
    hf_urls = []
    for file in files:
        if fnmatch.fnmatch(file, cfg.hf_url_glob):
            hf_url = construct_hf_url(cfg.hf_dataset_id, cfg.revision, file)
            hf_urls.append(hf_url)

    # Download entire dataset snapshot
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Downloading dataset {cfg.hf_dataset_id} (revision: {cfg.revision}) to temporary directory {temp_dir}.")
        local_dataset_path = snapshot_download(
            repo_id=cfg.hf_dataset_id,
            revision=cfg.revision,
            repo_type="dataset",
            local_dir=temp_dir,
            token=hf_token
        )

        # Upload the entire dataset directory to GCS
        for root, _, files in os.walk(local_dataset_path):
            for file in files:
                if fnmatch.fnmatch(file, cfg.hf_url_glob):
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_dataset_path)
                    gcs_file_path = os.path.join(gcs_versioned_relative_path, relative_path)

                    # Upload file to GCS
                    blob = bucket.blob(gcs_file_path)
                    blob.upload_from_filename(local_file_path)
                    logging.info(f"Uploaded {local_file_path} to GCS path: {gcs_file_path}")
                
    # Write Provenance JSON
    write_provenance_json(
        Path(gcs_versioned_relative_path),
        gcs_bucket,
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": hf_urls},
    )

    print(f"Uploaded all files and wrote provenance JSON; check {cfg.gcs_output_path}.")

@draccus.wrap()
def download_gated_main(cfg: DownloadConfig):
    download_and_upload_to_gcs(cfg)

if __name__ == "__main__":
    download_gated_main()