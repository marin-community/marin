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
Transform any HuggingFace preference dataset (for DPO, etc) to a standard format within each preference "column".

Usage:
- Register your adapter in preference_data_adapters.py
- Run this script with TransformPreferenceDatasetConfig.

Example:
uv run zephyr --backend=ray --max-parallelism=100 --memory=8GB \
    lib/marin/src/marin/transform/conversation/transform_preference_data.py \
    --input_path gs://bucket/path/to/dataset \
    --output_path gs://bucket/output/path \
    --source HuggingFaceH4/ultrafeedback_binarized
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field

import datasets
import draccus
from datasets import get_dataset_config_info
from zephyr import Dataset, flow_backend, write_jsonl_file

from .preference_data_adapters import PreferenceTransformAdapter, get_preference_adapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformPreferenceDatasetConfig:
    """Configuration for transforming a preference dataset from huggingface json/parquet to OpenAI format."""

    input_path: str
    output_path: str
    metadata_columns: list[str]
    source: str
    adapter_name: str
    filetype: str = "parquet"
    shard_size: int = 10000
    subsets: list[str] = field(default_factory=lambda: [])
    splits: list[str] = field(default_factory=lambda: ["train"])


@dataclass(frozen=True)
class SplitTask:
    """Task for processing a single subset/split combination."""

    input_path: str  # GCS or local path
    subset: str
    split: str
    output_path: str
    cfg: "TransformPreferenceDatasetConfig"


def generate_hash_from_pair(chosen, rejected) -> str:
    """Generate a hash from chosen and rejected message lists."""
    return hashlib.sha256((str(chosen) + str(rejected)).encode()).hexdigest()


def transform_row(row: dict, cfg: TransformPreferenceDatasetConfig, adapter: PreferenceTransformAdapter):
    example = adapter.extract_preference_example(row)
    if example is None:
        return None
    chosen_dicts = [msg.__dict__ for msg in example["chosen"]]
    rejected_dicts = [msg.__dict__ for msg in example["rejected"]]
    result = {
        "chosen": chosen_dicts,
        "rejected": rejected_dicts,
        "hash": generate_hash_from_pair(chosen_dicts, rejected_dicts),
    }
    for col in cfg.metadata_columns:
        if col in row:
            result[col] = row[col]
    return result


def get_shard_dir(dir_name: str, subset_name: str | None, split: str) -> str:
    if (subset_name == "default") or (subset_name is None):
        return os.path.join(dir_name, split)

    logger.info(f"Getting shard dir for {dir_name} {subset_name} {split}")
    logger.info(f"shard dir (os.path.join(dir_name, subset_name, split)): {os.path.join(dir_name, subset_name, split)}")
    return os.path.join(dir_name, subset_name, split)


def get_dataset_tasks(cfg: TransformPreferenceDatasetConfig):
    """Identify all subset/split combinations to process.

    Yields SplitTask objects for each subset/split combination.
    """
    input_path = cfg.input_path

    # 1. Identify subsets
    if cfg.subsets:
        subsets = cfg.subsets
    else:
        subsets = list(datasets.get_dataset_infos(path=input_path).keys())

    # 2. For each subset, get the splits
    for subset in subsets:
        config_info = get_dataset_config_info(input_path, config_name=subset)
        available_splits = list(config_info.splits.keys())

        if cfg.splits:
            splits = cfg.splits
            extra_splits = set(splits) - set(available_splits)
            if extra_splits:
                logger.warning(f"Requested split(s) {extra_splits} for {cfg.source} skipped.")
                splits = [s for s in splits if s in available_splits]
        else:
            splits = available_splits

        # Yield a task for each subset/split combination
        for split in splits:
            subset_output_path = get_shard_dir(cfg.output_path, subset, split)
            yield SplitTask(
                input_path=input_path,
                subset=subset,
                split=split,
                output_path=subset_output_path,
                cfg=cfg,
            )


def process_split_task(task: SplitTask) -> dict:
    """Load a subset/split, transform records, and write to output shards.

    This function streams records through transformation and writes them to
    multiple shard files in the task's output directory, maintaining the
    subset/split hierarchy.

    Args:
        task: SplitTask with input_path, subset, split, output_path, and cfg

    Returns:
        Dict with subset, split, shards (list of file metadata), and total_count
    """
    subset = task.subset
    split = task.split
    output_path = task.output_path
    cfg = task.cfg

    adapter = get_preference_adapter(cfg.adapter_name or cfg.source)
    if adapter is None:
        raise ValueError(f"No preference adapter found for source: {cfg.adapter_name or cfg.source}")

    logger.info(f"Processing subset: {subset}, split: {split}")
    dataset = datasets.load_dataset(path=task.input_path, name=subset, split=split, streaming=True)

    # Batch records and write to multiple shard files
    shard_files = []
    shard_idx = 0
    batch = []

    for row in dataset:
        transformed_row = transform_row(row, cfg, adapter)
        if transformed_row is not None:
            batch.append(transformed_row)

            if len(batch) >= cfg.shard_size:
                # Write this shard
                output_file = f"{output_path}/shard-{shard_idx:05d}.jsonl.gz"
                result = write_jsonl_file(batch, output_file)
                shard_files.append(result)
                shard_idx += 1
                batch = []

    # Write remaining records
    if batch:
        output_file = f"{output_path}/shard-{shard_idx:05d}.jsonl.gz"
        result = write_jsonl_file(batch, output_file)
        shard_files.append(result)

    total_count = sum(f["count"] for f in shard_files)
    return {
        "subset": subset,
        "split": split,
        "shards": shard_files,
        "total_count": total_count,
    }


@draccus.wrap()
def transform_hf_preference_dataset(cfg: TransformPreferenceDatasetConfig):
    """Transform HuggingFace preference dataset using task-level parallelism.

    All subset/split combinations are processed in parallel. Each task loads,
    transforms, and writes its records to maintain the directory structure.
    """
    backend = flow_backend()

    # Get all tasks (subset/split combinations)
    tasks = list(get_dataset_tasks(cfg))
    logger.info(f"Processing {len(tasks)} subset/split combinations")

    # Process all tasks in parallel
    pipeline = Dataset.from_list(tasks).map(process_split_task)
    results = list(backend.execute(pipeline))

    # Log summary
    for result in results:
        logger.info(
            f"Wrote {result['total_count']} records to {len(result['shards'])} shards "
            f"({result['subset']}/{result['split']})"
        )
        for shard in result["shards"]:
            logger.info(f"  - {shard['path']}: {shard['count']} records")
