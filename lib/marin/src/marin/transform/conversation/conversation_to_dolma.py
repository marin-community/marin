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
from zephyr import Backend, Dataset, load_jsonl


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
    Backend.execute(pipeline)


# Alias for other callers
convert_conversation_to_dolma = process_dataset


if __name__ == "__main__":
    process_dataset = draccus.wrap(process_dataset)
    process_dataset()
