"""
This script transforms the OpenHermes-2.5 dataset from huggingface json to OpenAI messages formatted jsonl.

The text in the dataset is a list of conversations, where each conversation is a list of messages
alternating between user and assistant. We combine each conversation into a single string by
concatenating all the messages and separating them with a space delimiter.

Usage Examples:
1. Register your adapter in adapters.py
2. Run the script as shown below with the same adapter name as the source you registered.

MetamathQA:
ray job submit --working-dir . -- python -m operations.transform.conversation.transform_conversation \
     --input_path gs://marin-us-central2/raw/metamathqa/aa4f34d/huggingface.co/datasets/ \
     --output_path gs://marin-us-central2/documents/metamathqa \
     --source meta-math/MetaMathQA \
     --metadata_columns ["type"] \
     --filetype json

Tulu-V2-SFT-Mixture:
ray job submit --working-dir . -- python -m operations.transform.conversation.transform_conversation \
     --input_path gs://marin-us-central2/raw/tulu-v2-sft-mixture/6248b17/huggingface.co/datasets/ \
     --output_path gs://marin-us-central2/documents/tulu-v2-sft-mixture \
     --source allenai/tulu-v2-sft-mixture \
     --metadata_columns '["dataset", "id"]' \
     --filetype parquet

UltraInteract_sft:
ray job submit --working-dir . -- python -m operations.transform.conversation.transform_conversation \
     --input_path gs://marin-us-central2/raw/ultrainteract-sft/2b102e4/huggingface.co/datasets/ \
     --output_path gs://marin-us-central2/documents/ultrainteract-sft \
     --source openbmb/UltraInteract_sft \
     --metadata_columns '["task", "dataset"]' \
     --filetype parquet
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import draccus
import fsspec
import pandas as pd
import ray

from marin.core.runtime import TaskConfig, cached_or_construct_output, fsspec_mkdirs, map_files_in_directory

from .adapters import OpenAIChatMessage, TransformAdapter, get_adapter

logger = logging.getLogger("ray")


@dataclass
class TransformDatasetConfig:
    """Base configuration to transform a conversation dataset from huggingface json to OpenAI format.

    Args:
        input_path (str): The path to the input file.
        output_path (str): The path to the output file.
        shard_size (int): The number of rows per shard.
        source (str): The name of the HuggingFace dataset.
        conversation_column (str): The column in the json containing the conversation.
        role_key (str): The key in the conversation dictionary that contains the role of the message.
        user_value (str): The value in the conversation dictionary that indicates the user role.
        assistant_value (str): The value in the conversation dictionary that indicates the assistant role.
        system_value (str): The value in the conversation dictionary that indicates the system role.
        content_key (str): The key in the conversation dictionary that contains the content of the message.
        metadata_columns (list[str]): The columns to include in the metadata.

    Example of role_key, user_value, assistant_value, and system_value:
    In OpenHermes-2.5, a conversation can look like this:
    [ { "from": "human", "value": "..."},
      { "from": "gpt", "value": "..."} ]

    In this example, the role_key is "from", the user_value is "human", the assistant_value is "gpt",
    and the system_value is "system". This helps us map the roles to the correct values in the OpenAI
    format from "from" -> "role" and "human"/"gpt" -> "user"/"assistant".
    """

    input_path: str = (
        "gs://marin-us-central2/raw/teknium--OpenHermes-2.5/b820378/huggingface.co/datasets/teknium/OpenHermes-2.5/resolve/b820378/openhermes2_5.json"
    )
    output_path: str = "gs://marin-us-central2/documents/teknium--OpenHermes-2.5/v2024-09-29"
    shard_size: int = 5000
    metadata_columns: list[str] = field(default_factory=lambda: ["source", "category", "skip_prompt_formatting"])
    source: str = "teknium/OpenHermes-2.5"
    filetype: str = "json"


def generate_hash_from_messages(messages: list[dict[str, str]]) -> str:
    """Generate a hash from a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages.

    Returns:
        str: A hash of the messages.
    """
    return hashlib.sha256(str(messages).encode()).hexdigest()


def transform_row(row: dict, cfg: TransformDatasetConfig, adapter: TransformAdapter):
    transformed_row_messages: list[OpenAIChatMessage] = adapter.transform_conversation_to_openai_format(row)
    transformed_row_messages = [message.dict() for message in transformed_row_messages]

    # Create a unique ID for the row based on the text
    row_idx = generate_hash_from_messages(transformed_row_messages)

    metadata = {col: row.get(col, "") for col in cfg.metadata_columns}
    return {
        "id": row_idx,
        "source": cfg.source,
        "messages": transformed_row_messages,
        "added": datetime.now(timezone.utc).isoformat(),
        "created": "",  # Not available in the dataset
        "metadata": metadata,
    }


def transform_rows(rows: list[dict], cfg: TransformDatasetConfig):
    """Transform a list of rows from the OpenHermes-2.5 dataset to a list of dolma formatted jsonl rows.

    Args:
        rows (list[dict]): A list of rows from the OpenHermes-2.5 dataset.

    Returns:
        list[dict]: A list of dolma formatted jsonl rows.
    """
    transformed_rows = []
    adapter = get_adapter(cfg.source)
    for row in rows:
        transformed_row = transform_row(row, cfg, adapter)
        transformed_rows.append(transformed_row)

    return transformed_rows


def load_dataset(input_path: str) -> list[dict]:
    """Load a list of rows from the file. Currently supports jsonl, json, and parquet.

    Args:
        input_path (str): The path to the input file.

    Returns:
        list[dict]: A list of rows from the input file.
    """
    if input_path.endswith(".jsonl"):
        with fsspec.open(input_path, "rt") as f:
            return [json.loads(line) for line in f]
    elif input_path.endswith(".json"):
        with fsspec.open(input_path, "rt") as f:
            return json.load(f)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path, engine="pyarrow")
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file type: {input_path}")


def create_shard_output_directory(output_filename: str) -> str:
    """Given an output filename, create a directory for the shards.

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

    output_path = f"{protocol}://{path_without_suffix}"
    fsspec_mkdirs(output_path)
    return output_path


@ray.remote(memory=4 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def transform_dataset(input_filename: str, output_filename: str, cfg: TransformDatasetConfig):
    rows = load_dataset(input_filename)
    logger.info(f"Transforming {len(rows)} rows from {input_filename} to {output_filename}")

    output_path = create_shard_output_directory(output_filename)

    for idx, shard in enumerate(range(0, len(rows), cfg.shard_size)):
        shard_rows = rows[shard : min(shard + cfg.shard_size, len(rows))]
        shard_filename = os.path.join(output_path, f"shard_{idx:05d}.jsonl.gz")
        logger.info(f"Writing shard {idx} to {shard_filename}")
        with fsspec.open(shard_filename, "wt", compression="gzip") as f:
            transformed_shard_rows = transform_rows(shard_rows, cfg)
            for row in transformed_shard_rows:
                f.write(f"{json.dumps(row)}\n")


@draccus.wrap()
def main(cfg: TransformDatasetConfig):
    responses = map_files_in_directory(
        transform_dataset.remote, cfg.input_path, f"**/*.{cfg.filetype}", cfg.output_path, TaskConfig(), False, cfg
    )

    ray.get(responses)


if __name__ == "__main__":
    main()
