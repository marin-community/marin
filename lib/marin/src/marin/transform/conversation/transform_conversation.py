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
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import draccus
import fsspec
from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.execution import unwrap_versioned_value
from marin.utils import fsspec_mkdirs, load_dataset_with_backoff
from zephyr import Dataset, flow_backend, load_jsonl, write_jsonl_file

from .adapters import TransformAdapter

_RESERVED_TOP_LEVEL_FIELDS = {"id", "source", "messages", "added", "created", "metadata"}
DEFAULT_TEXT_REPLACEMENTS = {"<think>": "<|start_think|>", "</think>": "<|end_think|>"}

logger = logging.getLogger(__name__)


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
        max_parallelism (int | None): Maximum number of concurrent shard processing tasks.
            Set to lower values to avoid HF rate limits. Set to None for default behavior (full concurrency).
    """

    source: str
    revision: str
    output_path: str
    metadata_columns: list[str]
    adapter: TransformAdapter
    subsets: list[str] = field(default_factory=lambda: [])  # Default behavior is to use all subsets
    splits: list[str] = field(default_factory=lambda: ["train"])  # Set to train; empty set means everything
    max_parallelism: int | None = None  # None means use default behavior (full concurrency)


@dataclass(frozen=True)
class ShardTask:
    """Task for processing a single shard of a dataset subset/split."""

    source: str  # HuggingFace dataset ID
    revision: str
    subset: str | None
    split: str
    shard_idx: int
    num_shards: int
    output_path: str
    cfg: TransformSFTDatasetConfig


def generate_hash_from_messages(messages: list[dict[str, str]]) -> str:
    """Generate a hash from a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages.

    Returns:
        str: A hash of the messages.
    """
    return hashlib.sha256(str(messages).encode()).hexdigest()


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def _normalize_tool_structures(message: dict) -> dict:
    tool_calls = message.get("tool_calls")
    if tool_calls:
        normalized_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            call_dict = dict(call)
            function = call_dict.get("function")
            if isinstance(function, dict):
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
            normalized_calls.append(call_dict)
        message["tool_calls"] = normalized_calls

    return message


def transform_row(row: dict, cfg: TransformSFTDatasetConfig, adapter: TransformAdapter):
    source = unwrap_versioned_value(cfg.source)
    transformed_row_messages: list[OpenAIChatMessage] = adapter.transform_conversation_to_openai_format(row)

    if transformed_row_messages is None:
        logger.warning(f"{source} returning no valid messages")
        return None

    transformed_row_messages = [message.model_dump() for message in transformed_row_messages]

    # Create a unique ID for the row based on the text
    row_idx = generate_hash_from_messages(transformed_row_messages)
    metadata_columns = unwrap_versioned_value(cfg.metadata_columns)
    metadata_remap = adapter.metadata_remap or {}
    replacements = adapter.replacements if adapter.replacements is not None else DEFAULT_TEXT_REPLACEMENTS

    metadata = {col: row.get(col, "") for col in metadata_columns}
    extra_columns: dict[str, object] = {}
    for source_column, target_column in metadata_remap.items():
        if target_column in _RESERVED_TOP_LEVEL_FIELDS:
            logging.log(
                logging.WARNING,
                f"Skipping remap for column '{source_column}' because target '{target_column}' is reserved.",
            )
            continue
        if source_column in row:
            extra_columns[target_column] = row[source_column]

    if replacements:
        for message in transformed_row_messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = _apply_replacements(content, replacements)
        transformed_row_messages = [_normalize_tool_structures(message) for message in transformed_row_messages]
    else:
        transformed_row_messages = [_normalize_tool_structures(message) for message in transformed_row_messages]
    if adapter.extra_metadata_fn:
        extra_from_fn = adapter.extra_metadata_fn(row)
        if extra_from_fn:
            extra_columns.update(extra_from_fn)
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


def _get_available_subsets(cfg: TransformSFTDatasetConfig) -> Sequence[str | None]:
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


def get_shard_dir(dir_name: os.PathLike, subset_name: str | None, split: str) -> os.PathLike | str:
    """Creates a new path with the subset and split names.
    e.g., create_subset_name('gs://thisserver/testfolder-a982374', 'subset', 'train') -> 'gs://thisserver/testfolder-a982374/subset/train'
    """
    if (subset_name == "default") or (subset_name is None):
        return os.path.join(dir_name, split)
    return os.path.join(dir_name, subset_name, split)


def get_dataset_tasks(cfg: TransformSFTDatasetConfig):
    """Identify all subset/split/shard combinations to process.

    Yields ShardTask objects for each shard of each subset/split combination.
    """
    source = unwrap_versioned_value(cfg.source)
    if not source:
        raise ValueError("Transform configuration must include `source` pointing to the HF dataset id.")
    revision = unwrap_versioned_value(cfg.revision)
    configured_splits = unwrap_versioned_value(cfg.splits)

    # 1. Get available subsets
    subsets = _get_available_subsets(cfg)
    if not subsets:
        raise ValueError(f"No subsets available for dataset {source}")

    # 2. For each subset, get the splits and shards
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

        # 3. For each split, enumerate shards
        for split in splits:
            subset_name = subset or "default"
            subset_output_path = get_shard_dir(cfg.output_path, subset_name, split)
            output_path = create_shard_output_directory(subset_output_path)

            dataset_kwargs: dict[str, object] = {
                "path": source,
                "split": split,
                "streaming": True,
                "revision": revision,
            }
            if subset not in (None, "default"):
                dataset_kwargs["name"] = subset

            dataset = load_dataset_with_backoff(
                context=f"{source} subset={subset_name} split={split}",
                logger=logger,
                **dataset_kwargs,
            )
            num_shards = dataset.num_shards
            if not num_shards:
                raise ValueError(f"Streaming dataset {source} subset={subset} split={split} does not expose num_shards.")

            # Yield a task for each shard
            for shard_idx in range(num_shards):
                yield ShardTask(
                    source=source,
                    revision=revision,
                    subset=subset,
                    split=split,
                    shard_idx=shard_idx,
                    num_shards=num_shards,
                    output_path=output_path,
                    cfg=cfg,
                )


def process_shard_task(task: ShardTask) -> dict:
    """Process a single shard of a dataset subset/split.

    Loads a specific shard from HuggingFace Hub, transforms records, and writes to output file.
    """
    adapter = unwrap_versioned_value(task.cfg.adapter).copy()
    if adapter is None:
        raise ValueError("Transform configuration requires an adapter.")

    subset_name = task.subset or "default"
    output_filename = _shard_filename(task.output_path, task.shard_idx)

    # If output already exists, skip the work to let Zephyr resume cleanly without sentinels.
    fs, _ = fsspec.core.url_to_fs(output_filename)
    if fs.exists(output_filename):
        logging.info(
            f"Skipping subset={subset_name} split={task.split} shard={task.shard_idx} "
            f"because output exists: {output_filename}"
        )
        return {
            "subset": subset_name,
            "split": task.split,
            "shard_idx": task.shard_idx,
            "path": output_filename,
            "count": 0,
            "skipped": True,
        }

    dataset_kwargs: dict[str, object] = {
        "path": task.source,
        "split": task.split,
        "streaming": True,
        "revision": task.revision,
    }
    if task.subset not in (None, "default"):
        dataset_kwargs["name"] = task.subset

    dataset = load_dataset_with_backoff(
        context=f"{task.source} subset={subset_name} split={task.split} shard={task.shard_idx}",
        logger=logger,
        **dataset_kwargs,
    )
    shard_dataset = dataset.shard(num_shards=task.num_shards, index=task.shard_idx)

    def transform_records():
        """Generator that yields transformed records."""
        for raw_row in shard_dataset:
            transformed_row = transform_row(raw_row, task.cfg, adapter)
            if transformed_row is not None:
                yield transformed_row.model_dump()

    result = write_jsonl_file(transform_records(), output_filename)

    logging.info(
        f"Wrote {result['count']} rows to {result['path']} "
        f"for subset={subset_name} split={task.split} shard={task.shard_idx}"
    )

    return {
        "subset": subset_name,
        "split": task.split,
        "shard_idx": task.shard_idx,
        "path": result["path"],
        "count": result["count"],
    }


@draccus.wrap()
def transform_hf_dataset(cfg: TransformSFTDatasetConfig):
    """Transform HuggingFace conversation dataset using shard-level parallelism.

    Streams dataset from HuggingFace Hub and processes each shard in parallel using Zephyr.
    Each shard is processed independently and written to a separate output file.
    Skips processing for shards with existing metrics files.
    """
    # Get max_parallelism from config
    max_parallelism = unwrap_versioned_value(cfg.max_parallelism)

    # Configure backend with concurrency limit if specified
    if max_parallelism is not None:
        logger.info(f"Processing with max_parallelism={max_parallelism} to avoid HF rate limits")
        backend = flow_backend(max_parallelism=max_parallelism)
    else:
        logger.info("Processing with default concurrency")
        backend = flow_backend()

    all_tasks = list(get_dataset_tasks(cfg))
    logger.info(f"Found {len(all_tasks)} total shards across all subset/split combinations")

    metrics_path = os.path.join(cfg.output_path, "metrics")
    pipeline = (
        Dataset.from_list(all_tasks)
        .map(process_shard_task)
        .write_jsonl(f"{metrics_path}/{{shard:05d}}-transform.jsonl", skip_existing=True)
    )
    metric_files = list(backend.execute(pipeline))

    # Log summary by subset/split
    from collections import defaultdict

    by_subset_split = defaultdict(list)
    for metric_file in metric_files:
        result = next(iter(load_jsonl(metric_file)))
        key = (result["subset"], result["split"])
        by_subset_split[key].append(result)

    for (subset, split), shard_results in sorted(by_subset_split.items()):
        total_count = sum(r["count"] for r in shard_results)
        logger.info(f"Wrote {total_count} records to {len(shard_results)} shards ({subset}/{split})")
        for shard in sorted(shard_results, key=lambda x: x["shard_idx"]):
            skipped_suffix = " (skipped)" if shard.get("skipped") else ""
            logger.info(f"  - {shard['path']}: {shard['count']} records (shard {shard['shard_idx']}){skipped_suffix}")

    return cfg.output_path


if __name__ == "__main__":
    transform_hf_dataset()
