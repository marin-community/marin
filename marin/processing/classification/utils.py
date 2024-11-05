"""
Usage:
python3 -m marin.processing.download_gcs_data --output-dir ~/data
"""

import json
import os
from typing import Any

import fsspec
import numpy as np
from google.cloud import storage
from huggingface_hub import hf_hub_download


def download_file(filename: str, output_path: str):
    output_filename = os.path.basename(filename).replace(".gz", "")
    output_file_path = os.path.join(output_path, output_filename)
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


def convert_floats(obj: np.float32 | np.float64 | float) -> Any:
    """Convert numpy floats to native python floats
    This is necessary because the json module cannot serialize numpy floats.
    """
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(element) for element in obj)
    elif not is_json_serializable(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return convert_floats(obj)

        # If the object is not serializable, convert it to a string as a catch-all
        return str(obj)

    return obj
