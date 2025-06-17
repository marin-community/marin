"""
Transform any HuggingFace preference dataset (for DPO, etc) to a standard OpenAI chat format within each preference "column".

Usage:
- Register your adapter in preference_data_adapters.py
- Run this script with TransformPreferenceDatasetConfig.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import datasets
import draccus
import fsspec
import ray
from google.cloud import storage

from marin.core.conversation import OpenAIChatMessage
from marin.core.runtime import fsspec_mkdirs

from .preference_data_adapters import PreferenceTransformAdapter, get_preference_adapter

logger = logging.getLogger("ray")

@dataclass(frozen=True)
class TransformPreferenceDatasetConfig:
    """Configuration for transforming a preference dataset from huggingface json/parquet to OpenAI format."""
    input_path: str
    output_path: str
    shard_size: int
    metadata_columns: list[str]
    source: str
    filetype: str
    adapter_name: str
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])

def generate_hash_from_pair(chosen: list[dict], rejected: list[dict]) -> str:
    return hashlib.sha256((str(chosen) + str(rejected)).encode()).hexdigest()

def transform_row(row: dict, cfg: TransformPreferenceDatasetConfig, adapter: PreferenceTransformAdapter):
    example = adapter.extract_preference_example(row)
    if example is None:
        return None
    result = {
        "chosen": [msg.__dict__ for msg in example["chosen"]],
        "rejected": [msg.__dict__ for msg in example["rejected"]],
        "hash": generate_hash_from_pair(example["chosen"], example["rejected"]),
    }
    for col in cfg.metadata_columns:
        if col in row:
            result[col] = row[col]
    return result

def transform_rows(rows: list[dict], cfg: TransformPreferenceDatasetConfig, adapter: PreferenceTransformAdapter):
    transformed = []
    for row in rows:
        out = transform_row(row, cfg, adapter)
        if out is not None:
            transformed.append(out)
    return transformed

def create_shard_output_directory(output_filename: str) -> str:
    # Remove file suffix and create directory for shards
    if output_filename.endswith(".jsonl.gz"):
        output_path = output_filename[:-9]
    elif output_filename.endswith(".jsonl"):
        output_path = output_filename[:-6]
    elif output_filename.endswith(".json"):
        output_path = output_filename[:-5]
    else:
        output_path = output_filename
    fsspec_mkdirs(output_path)
    return output_path

def download_directory_from_gcs(bucket_name: str, gcs_directory_path: str, local_directory_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_directory_path)
    os.makedirs(local_directory_path, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith("provenance.json"):
            continue
        rel_path = os.path.relpath(blob.name, gcs_directory_path)
        target_path = os.path.join(local_directory_path, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        blob.download_to_filename(target_path)

def copy_dataset_from_gcp_to_local(input_gcp_path: os.PathLike):
    if input_gcp_path.startswith("gs://"):
        parsed_url = urlparse(input_gcp_path)
        bucket = parsed_url.netloc
        gcp_path = parsed_url.path.lstrip("/")
        dir_name = os.path.basename(gcp_path)
        download_directory_from_gcs(bucket, gcp_path, dir_name)
        input_path = dir_name
    else:
        raise Exception("Input is not a GCP path")
    return input_path

def get_shard_dir(dir_name: os.PathLike, subset_name: str | None, split: str) -> os.PathLike:
    if (subset_name == "default") or (subset_name is None):
        return os.path.join(dir_name, split)
    return os.path.join(dir_name, subset_name, split)

@ray.remote
def transform_hf_preference_dataset(cfg: TransformPreferenceDatasetConfig):
    # 1. Copy data from GCP to local instance
    local_data_dir = copy_dataset_from_gcp_to_local(cfg.input_path)
    # 2. Identify subsets
    if cfg.subsets:
        subsets = cfg.subsets
    else:
        subsets = [x for x in datasets.get_dataset_infos(path=local_data_dir)]
    adapter = get_preference_adapter(cfg.adapter_name or cfg.source)
    if adapter is None:
        raise ValueError(f"No preference adapter found for source: {cfg.adapter_name or cfg.source}")
    for subset in subsets:
        split_values = [x for x in datasets.get_dataset_infos(path=local_data_dir)[subset].splits.values()]
        if isinstance(split_values[0], dict):
            data_splits = [x["name"] for x in split_values]
        else:
            data_splits = [x.name for x in split_values]
        if cfg.splits:
            splits = cfg.splits
            extra_splits = list(set(splits).symmetric_difference(data_splits))
            if extra_splits:
                logging.log(logging.WARNING, f"Requested split(s) {extra_splits} for {cfg.source} skipped.")
                splits = list(set(splits).intersection(data_splits))
        else:
            splits = data_splits
        for split in splits:
            dataset = datasets.load_dataset(path=local_data_dir, name=subset, split=split)
            rows = [r for r in dataset]
            del dataset
            subset_output_path = get_shard_dir(cfg.output_path, subset, split)
            output_path = create_shard_output_directory(subset_output_path)
            for idx, shard in enumerate(range(0, len(rows), cfg.shard_size)):
                shard_rows = rows[shard : min(shard + cfg.shard_size, len(rows))]
                shard_filename = os.path.join(output_path, f"shard_{idx:05d}.jsonl.gz")
                logger.info(f"Writing shard {idx} to {shard_filename}")
                with fsspec.open(shard_filename, "wt", compression="gzip") as f:
                    transformed_shard_rows = transform_rows(shard_rows, cfg, adapter)
                    for row in transformed_shard_rows:
                        f.write(f"{json.dumps(row)}\n")
            logging.log(logging.INFO, f"Wrote processed data to {output_path}")
    return cfg.output_path

@draccus.wrap()
def main(cfg: TransformPreferenceDatasetConfig):
    ray.get(transform_hf_preference_dataset.remote(cfg))
