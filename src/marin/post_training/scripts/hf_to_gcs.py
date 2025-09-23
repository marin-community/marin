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
import tempfile
from pathlib import Path

import msgpack
import numpy as np
import torch
from google.cloud import storage
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from marin.post_training.flax.utils import save_checkpoint


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


def convert_hf_to_jax_key(hf_key: str) -> tuple:
    """Convert HuggingFace parameter key to JAX/Marin format."""
    if hf_key == "model.embed_tokens.weight":
        return ("transformer", "wte", "embedding")
    elif hf_key == "model.norm.weight":
        return ("transformer", "ln_f", "kernel")
    elif hf_key == "lm_head.weight":
        return ("lm_head", "kernel")
    elif hf_key.startswith("model.layers."):
        # Parse layer index and component
        parts = hf_key.split(".")
        layer_idx = parts[2]

        if "self_attn" in hf_key:
            if "q_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "attention", "wq", "kernel")
            elif "k_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "attention", "wk", "kernel")
            elif "v_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "attention", "wv", "kernel")
            elif "o_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "attention", "wo", "kernel")
        elif "mlp" in hf_key:
            if "gate_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "feed_forward", "w1", "kernel")
            elif "up_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "feed_forward", "w3", "kernel")
            elif "down_proj.weight" in hf_key:
                return ("transformer", "h", layer_idx, "feed_forward", "w2", "kernel")
        elif "input_layernorm.weight" in hf_key:
            return ("transformer", "h", layer_idx, "attention_norm", "kernel")
        elif "post_attention_layernorm.weight" in hf_key:
            return ("transformer", "h", layer_idx, "ffn_norm", "kernel")

    raise ValueError(f"Unknown HuggingFace key: {hf_key}")


def convert_torch_to_jax(state_dict: dict) -> dict:
    """Convert PyTorch state dict to JAX/numpy arrays with proper nested structure."""
    jax_state_dict = {}

    for hf_key, value in state_dict.items():
        # Convert torch tensor to numpy
        if isinstance(value, torch.Tensor):
            # Move to CPU and convert to numpy
            np_array = value.detach().cpu().numpy()
        else:
            np_array = value

        # Transpose linear layer weights (HF stores as (in, out), JAX expects (out, in))
        # But NOT embedding layers which are stored correctly in HF format
        if hf_key.endswith(".weight") and len(np_array.shape) == 2 and not hf_key.endswith("embed_tokens.weight"):
            np_array = np_array.T

        # Convert HuggingFace key to JAX nested structure
        jax_key_path = convert_hf_to_jax_key(hf_key)

        # Create nested dict structure
        current = jax_state_dict
        for part in jax_key_path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[jax_key_path[-1]] = np_array

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


def copy_file(local_path: Path, dest_path: str):
    """Copy file to destination (local or GCS)."""
    print(f"Copying {local_path} to {dest_path}...")

    if dest_path.startswith("gs://"):
        # Parse GCS path
        parts = dest_path[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS path: {dest_path}")

        bucket_name, blob_name = parts

        # Upload using google-cloud-storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(local_path)
        print(f"Uploaded to {dest_path}")
    else:
        # Local file copy
        import shutil
        from pathlib import Path

        dest_dir = Path(dest_path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest_path)
        print(f"Copied to {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to Levanter format and upload to GCS")
    parser.add_argument("model_name", help="HuggingFace model name (e.g., timinar/baby-llama-58m)")
    parser.add_argument(
        "--gcs-dir",
        help="GCS output path (e.g., gs://marin-us-central2/rl_checkpoints/base/...)",
        type=str,
        required=True,
    )
    parser.add_argument("--model-type", default="llama", help="Model type for conversion (default: llama)")

    args = parser.parse_args()
    gcs_base = args.gcs_dir.rstrip("/")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, tokenizer_path, config_path = download_hf_model(args.model_name, tmpdir)

        # Load the model's state dict
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        state_dict = model.state_dict()

        # Step 2: Convert to JAX format
        print("Converting to JAX format...")
        jax_params = convert_torch_to_jax(state_dict)
        del state_dict

        print("Saving checkpoint...")
        save_checkpoint(jax_params, f"{gcs_base}/params.msgpack", float_dtype="bf16")
        print("Saving tokenizer...")
        copy_file(tokenizer_path / "tokenizer.json", f"{gcs_base}/tokenizer.json")
        print("Copying config...")
        copy_file(config_path, f"{gcs_base}/config.json")

        # For tokenizer, we'll just reference the HF model name in model_paths
        # (as done in exp1403_rl_math.py)
        print("\nModel successfully converted and uploaded!")
        print("\nYou can use these paths in your experiment:")
        print(f'  "params": "{gcs_base}/params.msgpack"')
        print(f'  "tokenizer": "{args.model_name}"')
        print(f'  "config": "{gcs_base}/config.json"')


if __name__ == "__main__":
    main()
