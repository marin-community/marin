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

"""Download, flatten, and tokenize the LIMA dataset for validation.

https://huggingface.co/datasets/GAIR/lima
"""

import os
from dataclasses import dataclass

from marin.download.huggingface.download_hf import DownloadConfig as HfDownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import (
    deferred,
    executor_main,
    output,
    step,
    versioned,
)
from zephyr import Backend, Dataset, load_jsonl

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

# Mark library functions as deferred
download_hf = deferred(_download_hf)


@step(name="raw/lima")
def lima():
    return download_hf(
        HfDownloadConfig(
            hf_dataset_id=versioned("GAIR/lima"),
            revision=versioned("68958e9"),
            gcs_output_path=output(),
            hf_urls_glob=["*.jsonl"],
            wait_for_completion=True,
        )
    )


@dataclass
class LimaConversationsToTextConfig:
    """Configuration for converting LIMA conversations to plain text."""

    raw_lima: str
    output_path: str


def convert_to_text(record):
    """Convert LIMA conversation format to plain text."""
    parts = []
    for i, content in enumerate(record.get("conversations", [])):
        role = "User" if i % 2 == 0 else "Assistant"
        parts.append(f"{role}: {content}")

    if parts:
        return {"text": "\n\n".join(parts)}
    return None


def _convert_lima_conversations(config: LimaConversationsToTextConfig):
    input_path = os.path.join(config.raw_lima, "train.jsonl")
    output_path = os.path.join(config.output_path, "train.jsonl")

    pipeline = (
        Dataset.from_files(input_path)
        .flat_map(load_jsonl)
        .map(convert_to_text)
        .filter(lambda x: x is not None)
        .write_jsonl(output_path)
    )
    Backend.execute(pipeline)


convert_lima_conversations = deferred(_convert_lima_conversations)


@step(name="raw/lima_text")
def lima_text():
    return convert_lima_conversations(
        LimaConversationsToTextConfig(
            raw_lima=lima(),
            output_path=output(),
        )
    )


@step(name="lima/tokenized")
def lima_tokenized(tokenizer: str = llama3_tokenizer, is_validation: bool = True):
    """Tokenize the LIMA validation set using both train and test splits."""
    return default_tokenize(
        name="lima_text",
        dataset=lima_text().with_output_path("raw/lima_text-68958e9/68958e9").cd("train.jsonl"),
        tokenizer=tokenizer,
        is_validation=is_validation,
    )


if __name__ == "__main__":
    executor_main(steps=[lima_tokenized()])
