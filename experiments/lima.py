"""Download, flatten, and tokenize the LIMA dataset for validation.

https://huggingface.co/datasets/GAIR/lima
"""

import json
from dataclasses import dataclass
from experiments.defaults import default_tokenize

import fsspec
import ray
import os

from marin.core.runtime import cached_or_construct_output
from marin.download import HfDownloadConfig, download_hf_gated_manual
from experiments.llama import llama3_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.processing.tokenize.data_configs import TokenizerStep

lima = (
    ExecutorStep(
        name="raw/lima",
        fn=download_hf_gated_manual,
        config=HfDownloadConfig(
            hf_dataset_id=versioned("GAIR/lima"),
            revision=versioned("68958e9"),
            gcs_output_path=this_output_path(),
            hf_urls_glob=["*.jsonl"],
            wait_for_completion=True,
        ),
    )
    .with_output_path("raw/lima-68958e9")
    .cd("68958e9")
)


@dataclass
class LimaConversationsToTextConfig:
    """Configuration for converting LIMA conversations to plain text."""

    raw_lima: str


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _convert_lima_dataset(input_path, output_path):
    """Convert a single LIMA jsonl file with conversations to a plain text jsonl file."""
    output_json = os.path.join(output_path, "train.jsonl")
    with fsspec.open(input_path, "rt") as src, fsspec.open(output_json, "wt") as dst:
        for line in src:
            record = json.loads(line)
            parts = []
            for i, content in enumerate(record.get("conversations", [])):
                if i == 0:
                    print(content)
                role = "User" if i % 2 == 0 else "Assistant"
                parts.append(f"{role}: {content}")
            if parts:
                dst.write(json.dumps({"text": "\n\n".join(parts)}) + "\n")
    return True


def convert_lima_conversations(config: LimaConversationsToTextConfig):
    input_path = os.path.join(config.raw_lima, "train.jsonl")
    future = _convert_lima_dataset.remote(input_path, config.raw_lima.replace("lima", "lima_text"))

    ray.get(future)


lima_text = ExecutorStep(
    name="raw/lima_text",
    fn=convert_lima_conversations,
    config=LimaConversationsToTextConfig(raw_lima=lima),
).with_output_path("raw/lima_text-68958e9/68958e9")


def lima_tokenized(tokenizer: str = llama3_tokenizer, is_validation: bool = True) -> dict[str, TokenizerStep]:
    """Return steps to tokenize the LIMA validation set using both train and test splits."""

    return default_tokenize(
        name="lima_text",
        dataset=lima_text.cd("train.jsonl"),
        tokenizer=tokenizer,
        is_validation=is_validation,
    )


if __name__ == "__main__":
    executor_main(steps=[lima_tokenized()])
