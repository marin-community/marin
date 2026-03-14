# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Transform conversation format to Dolma format.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --cluster=us-central2 \
    lib/marin/src/marin/transform/conversation/conversation_to_dolma.py \
    --input_path gs://marin-us-central2/conversations/ \
    --output_path gs://marin-data/processed/conversations/dolma-v1.0/
"""

import dataclasses

import draccus
from marin.execution.executor import THIS_OUTPUT_PATH
from zephyr import Dataset, ZephyrContext, load_jsonl


@dataclasses.dataclass
class ConversationToDolmaConfig:
    input_path: str
    output_path: str = THIS_OUTPUT_PATH


def transform_conversation_to_dolma(row: dict):
    dolma_row = row
    text = ""
    for message in dolma_row["messages"]:
        text += message["role"] + ": "
        text += message["content"] + "\n\n"

    text = text.strip()

    dolma_row["text"] = text
    del dolma_row["messages"]
    return dolma_row


def process_dataset(config: ConversationToDolmaConfig):
    """Convert conversation format to Dolma format."""
    pipeline = (
        Dataset.from_files(f"{config.input_path}/**/*.jsonl.gz", empty_glob_ok=False)
        .flat_map(load_jsonl)
        .map(transform_conversation_to_dolma)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    with ZephyrContext(name="conversation-to-dolma") as ctx:
        ctx.execute(pipeline)


# Alias for other callers
convert_conversation_to_dolma = process_dataset


if __name__ == "__main__":
    process_dataset = draccus.wrap(process_dataset)
    process_dataset()
