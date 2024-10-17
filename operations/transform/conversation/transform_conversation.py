"""
This script transforms the OpenHermes-2.5 dataset from huggingface json to dolma formatted jsonl.

The text in the dataset is a list of conversations, where each conversation is a list of messages
alternating between user and assistant. We combine each conversation into a single string by
concatenating all the messages and separating them with a space delimiter.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import draccus
import fsspec
import ray

from .adapters import OpenAIChatMessage, get_adapter


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


def generate_hash_from_messages(messages: list[dict[str, str]]) -> str:
    """Generate a hash from a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages.

    Returns:
        str: A hash of the messages.
    """
    return hashlib.sha256(str(messages).encode()).hexdigest()


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
        transformed_row_messages: list[OpenAIChatMessage] = adapter.transform_conversation_to_openai_format(row)
        transformed_row_messages = [message.dict() for message in transformed_row_messages]

        # Create a unique ID for the row based on the text
        row_idx = generate_hash_from_messages(transformed_row_messages)

        metadata = {col: row.get(col, "") for col in cfg.metadata_columns}
        transformed_rows.append(
            {
                "id": row_idx,
                "source": cfg.source,
                "messages": transformed_row_messages,
                "added": datetime.now(timezone.utc).isoformat(),
                "created": "",  # Not available in the dataset
                "metadata": metadata,
            }
        )
    return transformed_rows


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def transform(cfg: TransformDatasetConfig):
    rows = []
    with fsspec.open(cfg.input_path, "rt") as f:
        rows = json.load(f)

    for idx, shard in enumerate(range(0, len(rows), cfg.shard_size)):
        shard_rows = rows[shard : shard + cfg.shard_size]
        shard_path = f"{cfg.output_path}/shard_{idx:05d}.jsonl.gz"
        with fsspec.open(shard_path, "wt", compression="gzip") as f:
            transformed_shard_rows = transform_rows(shard_rows, cfg)
            for row in transformed_shard_rows:
                f.write(f"{json.dumps(row)}\n")


@draccus.wrap()
def main(cfg: TransformDatasetConfig):
    response = transform.remote(cfg)

    try:
        ray.get(response)
    except Exception as e:
        print("Error processing OpenHermes-2.5 dataset: {e}")
        raise e


if __name__ == "__main__":
    main()
