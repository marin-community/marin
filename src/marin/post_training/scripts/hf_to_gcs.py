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
Download HuggingFace models and writes them to GCS for use with marin.

Usage:
    uv run hf_to_gcs.py <hf_model_name> <gs_output_path> [--model-type TYPE]

Example:
    uv run hf_to_gcs.py timinar/baby-llama-58m gs://marin-us-central2/rl_checkpoints/base/baby-llama-58m
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import msgpack
import numpy as np
import torch
from google.cloud import storage
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def download_hf_model(model_name: str, cache_dir: str) -> tuple[Path, Path, Path]:
    """Download HuggingFace model and tokenizer.

    Returns:
        Tuple of (model_path, tokenizer_path, config_path)
    """
    print(f"Downloading model {model_name}...")

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer_path = Path(cache_dir) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)

    # Download config
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config_path = Path(cache_dir) / "config.json"
    config.save_pretrained(Path(cache_dir))

    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model_path = Path(cache_dir) / "model"
    model.save_pretrained(model_path)

    return model_path, tokenizer_path, config_path


def convert_torch_to_jax(state_dict: dict) -> dict:
    """Convert PyTorch state dict to JAX/numpy arrays."""
    jax_state_dict = {}

    for key, value in state_dict.items():
        # Convert torch tensor to numpy
        if isinstance(value, torch.Tensor):
            # Move to CPU and convert to numpy
            np_array = value.detach().cpu().numpy()
            jax_state_dict[key] = np_array
        else:
            jax_state_dict[key] = value

    return jax_state_dict


def save_msgpack(params: dict, output_path: Path):
    """Save parameters as msgpack file."""
    print(f"Saving params to {output_path}...")

    # Convert numpy arrays to lists for msgpack serialization
    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return type(obj)(numpy_to_list(item) for item in obj)
        else:
            return obj

    # Convert to msgpack format
    with open(output_path, "wb") as f:
        serializable_params = numpy_to_list(params)
        packed = msgpack.packb(serializable_params, use_bin_type=True)
        f.write(packed)


def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload a file to Google Cloud Storage."""
    print(f"Uploading {local_path} to {gcs_path}...")

    # Parse GCS path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"GCS path must start with gs://: {gcs_path}")

    parts = gcs_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    bucket_name, blob_name = parts

    # Upload using google-cloud-storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path)
    print(f"Uploaded to {gcs_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to Levanter format and upload to GCS")
    parser.add_argument("model_name", help="HuggingFace model name (e.g., timinar/baby-llama-58m)")
    parser.add_argument("gcs_path", help="GCS output path (e.g., gs://marin-us-central2/rl_checkpoints/base/...)")
    parser.add_argument("--model-type", default="llama", help="Model type for conversion (default: llama)")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for downloads")

    args = parser.parse_args()

    # Use a temporary directory if no cache dir specified
    if args.cache_dir:
        cache_dir = args.cache_dir
        cleanup = False
    else:
        cache_dir = tempfile.mkdtemp(prefix="hf_to_gcs_")
        cleanup = True

    try:
        # Step 1: Download HF model
        model_path, tokenizer_path, config_path = download_hf_model(args.model_name, cache_dir)

        # Load the model's state dict
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        state_dict = model.state_dict()

        # Step 2: Convert to JAX format
        print("Converting to JAX format...")
        jax_params = convert_torch_to_jax(state_dict)

        # Step 3: Save as msgpack
        params_file = Path(cache_dir) / "params.msgpack"
        save_msgpack(jax_params, params_file)

        # Step 4: Upload to GCS
        # Make sure the GCS path ends without trailing slash
        gcs_base = args.gcs_path.rstrip("/")

        # Upload params
        upload_to_gcs(params_file, f"{gcs_base}/params.msgpack")

        # Update tokenizer
        upload_to_gcs(tokenizer_path / "tokenizer.json", f"{gcs_base}/tokenizer.json")

        # Upload config
        upload_to_gcs(config_path, f"{gcs_base}/config.json")

        # For tokenizer, we'll just reference the HF model name in model_paths
        # (as done in exp1403_rl_math.py)
        print("\nModel successfully converted and uploaded!")
        print("\nYou can use these paths in your experiment:")
        print(f'  "params": "{gcs_base}/params.msgpack"')
        print(f'  "tokenizer": "{args.model_name}"')
        print(f'  "config": "{gcs_base}/config.json"')

    finally:
        if cleanup:
            print(f"Cleaning up temporary directory {cache_dir}")
            shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()
