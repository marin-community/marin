"""
This script transforms the OpenHermes-2.5 dataset from huggingface json to dolma formatted jsonl.

The text in the dataset is a list of conversations, where each conversation is a list of messages
alternating between user and assistant. We combine each conversation into a single string by
concatenating all the messages and separating them with a space delimiter.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import draccus
import fsspec
import ray


@dataclass
class TransformOpenhermesConfig:
    """Configuration to transform the OpenHermes-2.5 dataset from huggingface json to dolma formatted jsonl.

    Args:
        input_path (str): The path to the input file.
        output_path (str): The path to the output file.
        shard_size (int): The number of rows per shard.
    """

    input_path: str = (
        "gs://marin-us-central2/raw/teknium--OpenHermes-2.5/b820378/huggingface.co/datasets/teknium/OpenHermes-2.5/resolve/b820378/openhermes2_5.json"
    )
    output_path: str = "gs://marin-us-central2/documents/teknium--OpenHermes-2.5/v2024-09-29"
    shard_size: int = 5000


def transform_conversation(conversation: list[dict[str, str]]):
    """Transform a conversation from the OpenHermes-2.5 dataset to a string.

    Args:
        conversation (list[dict[str, str]]): A list of messages in the conversation.

    Returns:
        str: A string representation of the conversation created by concatenating all the messages
            with a space delimiter.
    """

    conv_string = ""
    for message in conversation:
        conv_string += f"{message['value']}" + " "

    return conv_string


def transform_rows(rows: list[dict]):
    """Transform a list of rows from the OpenHermes-2.5 dataset to a list of dolma formatted jsonl rows.

    Args:
        rows (list[dict]): A list of rows from the OpenHermes-2.5 dataset.

    Returns:
        list[dict]: A list of dolma formatted jsonl rows.
    """
    transformed_rows = []
    for row in rows:
        row_conversations: list[dict[str, str]] = row["conversations"]
        transformed_row_text = transform_conversation(row_conversations)

        # Create a unique ID for the row based on the text
        row_idx = hashlib.sha256(transformed_row_text.encode()).hexdigest()

        transformed_rows.append(
            {
                "id": row_idx,
                "text": transformed_row_text,
                "source": "teknium/OpenHermes-2.5",
                "added": datetime.now(timezone.utc).isoformat(),
                "created": "",  # Not available in the dataset
                "metadata": {
                    "source": row.get("source", ""),
                    "category": row.get("category", ""),
                    "skip_prompt_formatting": row.get("skip_prompt_formatting", False),
                },
            }
        )
    return transformed_rows


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def transform_openhermes(cfg: TransformOpenhermesConfig):
    rows = []
    with fsspec.open(cfg.input_path, "rt") as f:
        rows = json.load(f)

    for idx, shard in enumerate(range(0, len(rows), cfg.shard_size)):
        shard_rows = rows[shard : shard + cfg.shard_size]
        shard_path = f"{cfg.output_path}/shard_{idx:05d}.jsonl.gz"
        with fsspec.open(shard_path, "wt", compression="gzip") as f:
            transformed_shard_rows = transform_rows(shard_rows)
            for row in transformed_shard_rows:
                f.write(f"{json.dumps(row)}\n")


@draccus.wrap()
def main(cfg: TransformOpenhermesConfig):
    response = transform_openhermes.remote(cfg)

    try:
        ray.get(response)
    except Exception as e:
        print("Error processing OpenHermes-2.5 dataset: {e}")
        raise e


if __name__ == "__main__":
    main()
