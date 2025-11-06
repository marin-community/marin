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
Transform any HuggingFace dataset to OpenAI messages format.

Usage Instructions:
1. Register your adapter in adapters.py
2. Run the script, filling out the TransformSFTDatasetConfig.

Check out experiments/instruction_datasets.py to see how to run this using the Executor.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import datasets
import draccus
import fsspec
import ray
from tqdm_loggable.tqdm_logging import tqdm_logging

from marin.execution import unwrap_versioned_value
from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.core.runtime import fsspec_mkdirs

from .adapters import TransformAdapter

_RESERVED_TOP_LEVEL_FIELDS = {"id", "source", "messages", "added", "created", "metadata"}

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class TransformSFTDatasetConfig:
    """Base configuration to transform a conversation dataset from huggingface json to OpenAI format.

    Args:
        source (str): The name of the HuggingFace dataset.
        revision (str): The revision of the HuggingFace dataset to use.
        output_path (str): The base path where transformed shards will be written.
        metadata_columns (list[str]): Additional metadata keys to copy from the source row.
        adapter (TransformAdapter): Adapter responsible for mapping raw rows into OpenAI chat format.
        subsets (list[str]): Data subsets (from HuggingFace config) to use. Empty list indicates all/default subset(s).
        splits (list[str]): Data splits (e.g., `train`, `validation`) to use. Empty list indicates all splits.
        remap_columns (dict[str, str]): Mapping of source column names to new metadata keys preserved in the output.
    """

    source: str
    revision: str
    output_path: str
    metadata_columns: list[str]
    adapter: TransformAdapter
    subsets: list[str] = field(default_factory=lambda: [])  # Default behavior is to use all subsets
    splits: list[str] = field(default_factory=lambda: ["train"])  # Set to train; empty set means everything
    remap_columns: dict[str, str] = field(default_factory=dict)


def generate_hash_from_messages(messages: list[dict[str, str]]) -> str:
    """Generate a hash from a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages.

    Returns:
        str: A hash of the messages.
    """
    return hashlib.sha256(str(messages).encode()).hexdigest()


def transform_row(row: dict, cfg: TransformSFTDatasetConfig, adapter: TransformAdapter):
    transformed_row_messages: list[OpenAIChatMessage] = adapter.transform_conversation_to_openai_format(row)

    if transformed_row_messages is None:
        logger.warning(f"{cfg.source} returning no valid messages")
        return None

    transformed_row_messages = [message.model_dump() for message in transformed_row_messages]
    source = unwrap_versioned_value(cfg.source)

    # Create a unique ID for the row based on the text
    row_idx = generate_hash_from_messages(transformed_row_messages)
    metadata_columns = unwrap_versioned_value(cfg.metadata_columns)
    remap_columns = unwrap_versioned_value(cfg.remap_columns)

    metadata = {col: row.get(col, "") for col in metadata_columns}
    extra_columns: dict[str, object] = {}
    for source_column, target_column in remap_columns.items():
        if target_column in _RESERVED_TOP_LEVEL_FIELDS:
            logging.log(
                logging.WARNING,
                f"Skipping remap for column '{source_column}' because target '{target_column}' is reserved.",
            )
            continue
        if source_column in row:
            extra_columns[target_column] = row[source_column]
    return DolmaConversationOutput(
        id=row_idx,
        source=source,
        messages=transformed_row_messages,
        added=datetime.now(timezone.utc).isoformat(),
        created="",  # Not available in the dataset
        metadata=metadata,
        **extra_columns,
    )


def create_shard_output_directory(output_filename: str) -> str:
    """Given an output filename, remove the suffix of the filename and create a directory for the shards.

    Example:
        [A] output_filename = "gs://A/B.jsonl.gz" -> [B] output_path = "gs://A/B"

    Args:
        output_filename (str): The path to the output file.

    Returns:
        str: The path to the directory containing the shards.
    """
    _, path = fsspec.core.url_to_fs(output_filename)
    protocol = fsspec.core.split_protocol(output_filename)[0]

    path_without_suffix = Path(path)
    while path_without_suffix.suffix:
        path_without_suffix = path_without_suffix.with_suffix("")

    if protocol:
        output_path = f"{protocol}://{path_without_suffix}"
    else:
        output_path = str(path_without_suffix)
    fsspec_mkdirs(output_path)
    return output_path


def _get_available_subsets(cfg: TransformSFTDatasetConfig) -> list[str | None]:
    configured_subsets = unwrap_versioned_value(cfg.subsets)
    if configured_subsets:
        return configured_subsets
    try:
        subsets = datasets.get_dataset_config_names(cfg.source)
    except Exception as exc:
        logging.log(logging.WARNING, f"Unable to fetch dataset configs for {cfg.source}: {exc}")
        subsets = []
    if not subsets:
        return [None]
    return subsets


def _get_available_splits(cfg: TransformSFTDatasetConfig, subset: str | None) -> list[str]:
    configured_splits = unwrap_versioned_value(cfg.splits)
    if configured_splits:
        return list(configured_splits)
    try:
        split_names = datasets.get_dataset_split_names(cfg.source, name=subset)
    except Exception as exc:
        logging.log(logging.WARNING, f"Unable to fetch splits for {cfg.source} (subset={subset}): {exc}")
        split_names = ["train"]
    if not split_names:
        return ["train"]
    return [split for split in split_names if split not in ("validation", "test")]


def _shard_filename(output_path: str, shard_idx: int) -> str:
    return os.path.join(output_path, f"shard_{shard_idx:05d}.jsonl.gz")


def get_shard_dir(dir_name: os.PathLike, subset_name: str | None, split: str) -> os.PathLike:
    """Creates a new path with the subset and split names.
    e.g., create_subset_name('gs://thisserver/testfolder-a982374', 'subset', 'train') -> 'gs://thisserver/testfolder-a982374/subset/train'
    """
    if (subset_name == "default") or (subset_name is None):
        return os.path.join(dir_name, split)
    return os.path.join(dir_name, subset_name, split)


@ray.remote
def process_streaming_shard(
    cfg: TransformSFTDatasetConfig, subset: str | None, split: str, shard_idx: int, num_shards: int
):
    adapter = unwrap_versioned_value(cfg.adapter).copy()
    if adapter is None:
        raise ValueError("Transform configuration requires an adapter.")
    source = unwrap_versioned_value(cfg.source)
    if not source:
        raise ValueError("Transform configuration must include `source` pointing to the HF dataset id.")
    revision = unwrap_versioned_value(cfg.revision)
    dataset_kwargs: dict[str, object] = {
        "path": source,
        "split": split,
        "streaming": True,
        "revision": revision,
    }
    if subset not in (None, "default"):
        dataset_kwargs["name"] = subset

    dataset = datasets.load_dataset(**dataset_kwargs)
    shard_dataset = dataset.shard(num_shards=num_shards, index=shard_idx)

    subset_name = subset or "default"
    subset_output_path = get_shard_dir(cfg.output_path, subset_name, split)
    output_path = create_shard_output_directory(subset_output_path)
    rows_written = 0
    files_written: list[str] = []
    current_handle = None
    current_filename = None
    rows_in_current_file = 0

    tqdm_logging.log_level = logging.WARNING
    pbar = tqdm_logging(desc=f"Transforming {source} subset={subset_name} split={split} shard={shard_idx}")

    try:
        for raw_row in shard_dataset:
            transformed_row = transform_row(raw_row, cfg, adapter)
            if transformed_row is None:
                continue

            if current_handle is None:
                current_filename = _shard_filename(output_path, shard_idx)
                current_handle = fsspec.open(current_filename, "wt", compression="gzip").open()
                files_written.append(current_filename)
                rows_in_current_file = 0

            current_handle.write(f"{json.dumps(transformed_row.model_dump())}\n")
            rows_written += 1
            rows_in_current_file += 1
            pbar.update(1)
    finally:
        if current_handle is not None:
            current_handle.close()

    logging.info(
        f"Wrote {rows_written} rows to {current_filename} " f"for subset={subset_name} split={split} shard={shard_idx}"
    )
    return files_written


@ray.remote
def transform_hf_dataset(cfg: TransformSFTDatasetConfig):
    """Stream a HuggingFace dataset and write remote shards without local staging."""
    if cfg.adapter is None:
        raise ValueError("Transform configuration requires an adapter.")
    source = unwrap_versioned_value(cfg.source)
    if not source:
        raise ValueError("Transform configuration must include `source` pointing to the HF dataset id.")
    revision = unwrap_versioned_value(cfg.revision)
    configured_splits = unwrap_versioned_value(cfg.splits)
    subsets = _get_available_subsets(cfg)
    if not subsets:
        raise ValueError(f"No subsets available for dataset {source}")

    shard_refs = []

    for subset in subsets:
        splits = _get_available_splits(cfg, subset)
        if configured_splits:
            requested = set(configured_splits)
            missing = sorted(requested - set(splits))
            if missing:
                logging.log(logging.WARNING, f"Requested split(s) {missing} for {source} skipped.")
            splits = [split for split in splits if split in requested]
        if not splits:
            logging.log(logging.WARNING, f"No splits to process for subset={subset}; skipping.")
            continue

        for split in splits:
            dataset_kwargs: dict[str, object] = {
                "path": source,
                "split": split,
                "streaming": True,
                "revision": revision,
            }
            if subset not in (None, "default"):
                dataset_kwargs["name"] = subset

            dataset = datasets.load_dataset(**dataset_kwargs)
            num_shards = dataset.num_shards
            if not num_shards:
                raise ValueError(
                    f"Streaming dataset {source} subset={subset} split={split} does not expose num_shards."
                )

            for shard_idx in range(num_shards):
                shard_refs.append(process_streaming_shard.remote(cfg, subset, split, shard_idx, num_shards))

    ray.get(shard_refs)
    return cfg.output_path


@draccus.wrap()
def main(cfg: TransformSFTDatasetConfig):
    ray.get(transform_hf_dataset.remote(cfg))
