"""
Usage:
python3 -m marin.processing.download_gcs_data --output-dir ~/data
"""

import argparse
import os
import gzip
import shutil
import json

import fsspec
from google.cloud import storage
from typing import Any
from huggingface_hub import hf_hub_download


def download_file(filename: str, output_dir: str):
    output_filename = os.path.basename(filename).replace(".gz", "")
    output_file_path = os.path.join(output_dir, output_filename)
    file_format = os.path.basename(output_filename).split(".")[1]

    with fsspec.open(filename, "r", compression="gzip") as f:
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for line in f:
                if file_format == "json" or file_format == "jsonl":
                    json_line = json.loads(line)
                    f_out.write(json.dumps(json_line) + "\n")
                elif file_format == "txt":
                    f_out.write(line)


def download_huggingface_file_with_backoff(repo_id: str, filename: str, local_dir: str, output_path: str):
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    os.rename(os.path.join(local_dir, filename), output_path)


def download_gcs_file_with_backoff(bucket_name: str, blob_name: str, output_path: str):
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Download the model file
    blob = bucket.blob(blob_name)
    blob.download_to_filename(output_path)


def is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(element) for element in obj)
    elif not is_json_serializable(obj):
        return str(obj)
    return obj
