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

import dataclasses
import json

import draccus
import fsspec
import ray

from marin.core.runtime import TaskConfig, cached_or_construct_output, map_files_in_directory
from marin.execution.executor import THIS_OUTPUT_PATH


@dataclasses.dataclass
class ConversationToDolmaConfig:
    input_path: str
    output_path: str = THIS_OUTPUT_PATH


def role_to_str(role: str):
    if role == "user":
        return "U"
    elif role == "assistant":
        return "A"
    else:
        raise ValueError(f"Unknown role: {role}")


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


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename: str, output_filename: str):
    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
            for line in f_in:
                row = json.loads(line)
                dolma_row = transform_conversation_to_dolma(row)
                f_out.write(f"{json.dumps(dolma_row)}\n")


@ray.remote
def process_dataset(config: ConversationToDolmaConfig):
    responses = map_files_in_directory(
        process_file.remote, config.input_path, "**/*.jsonl.gz", config.output_path, TaskConfig(), False
    )
    ray.get(responses)


def convert_conversation_to_dolma(config: ConversationToDolmaConfig):
    ray.get(process_dataset.remote(config))


if __name__ == "__main__":
    draccus.wrap(convert_conversation_to_dolma)()
