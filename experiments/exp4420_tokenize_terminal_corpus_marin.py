# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize nvidia/Nemotron-Terminal-Corpus with the Marin tokenizer (CPU-only).

Run this first to build tokenized caches, then run the SFT training script
(exp4420_sft_marin_8b_instruct_terminal_corpus.py) which will skip tokenization.

Tracked in: https://github.com/marin-community/marin/issues/4420

Usage:
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --memory 16GB \
        -e MARIN_PREFIX gs://marin-us-central1 \
        -e HF_TOKEN ${HF_TOKEN} \
        --no-wait \
        --job-name kevin-exp4420-tokenize \
        -- python experiments/exp4420_tokenize_terminal_corpus_marin.py
"""

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.marin_models import MARIN_CHAT_TEMPLATE, marin_tokenizer
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from marin.execution.executor import executor_main

SUBSETS = [
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters",
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy",
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium",
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed",
]


def create_tokenization_step(dataset_identifier: str):
    short_name = dataset_identifier.split("/")[-1]
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE),
    )


steps = [create_tokenization_step(subset) for subset in SUBSETS]

if __name__ == "__main__":
    print(f"=== Tokenizing {len(SUBSETS)} Terminal-Corpus subsets with Marin tokenizer ===")
    for s in SUBSETS:
        print(f"  {s}")
    executor_main(steps=steps)
