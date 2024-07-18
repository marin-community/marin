"""
Usage:
python3 -m marin.processing.download_gcs_data --output-dir ~/data
"""

import argparse
import os
import gzip
import shutil
import fsspec
import json
from google.cloud import storage

# from tenacity import retry, stop_after_attempt, wait_exponential

from huggingface_hub import hf_hub_download


def download_file(filename, output_dir):
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


# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
def download_huggingface_file_with_backoff(repo_id, filename, local_dir, output_path):
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    os.rename(os.path.join(local_dir, filename), output_path)


# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
def download_gcs_file_with_backoff(bucket_name, blob_name, output_path):
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Download the model file
    blob = bucket.blob(blob_name)
    blob.download_to_filename(output_path)


def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(element) for element in obj)
    elif not is_json_serializable(obj):
        return str(obj)
    return obj
