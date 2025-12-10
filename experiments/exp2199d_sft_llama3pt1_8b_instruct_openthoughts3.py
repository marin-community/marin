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
Fine-tunes Llama-3.1-8b-Instruct (meta-llama/Llama-3.1-8B-Instruct) on the OpenThoughts3 dataset (open-thoughts/OpenThoughts3-1.2M).
"""
import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.llama import llama_3_1_8b_instruct, llama_3_1_8b_instruct_tokenizer
from experiments.qwen3 import qwen3_4b
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import lm_mixture_data_config

SLUGIFY_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = SLUGIFY_PATTERN.sub("_", value.lower()).strip("_")
    return slug or "dataset"


def build_dataset_specs() -> tuple[dict[str, str], dict[str, int]]:
    datasets: dict[str, str] = {}
    weights: dict[str, int] = {}
    datasets["openthoughts3"] = "open-thoughts/OpenThoughts3-1.2M"
    weights["openthoughts3"] = 1200000  # Has exactly 1.2M rows
    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str) -> ExecutorStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name}_llama_3_1_8b_instruct_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=llama_3_1_8b_instruct_tokenizer,
        format=ChatLmDatasetFormat(mask_user_turns=False),  # NOTE: Set to False since couldn't get an updated Llama-3.1 chat template with {%generation%} tags working
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = sum(mixture_weights.values())
TARGET_EPOCHS = 5
TRAIN_BATCH_SIZE = 512  # Fits on 1 x TPU-v5p-64 (for v5p-64, batch size must be divisible by # devices which is 32)
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

mixture_sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-64"),
    tokenizer=llama_3_1_8b_instruct_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=8e-5,
    max_seq_len=16384,
    seed=0,
    steps_per_checkpoint=1000,  # Around 2344 steps per epoch with batch size 512
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    permutation_type="feistel",
    shuffle=True,
    missing_weights_are_validation=True,
    mixture_block_size=12288,  # large block size to include the tiny datasets (namely s1k_1.1)
)

exp2199d_sft_llama_3_1_8b_instruct_openthoughts3 = default_sft(
    name="exp2199d_sft_llama_3_1_8b_instruct_openthoughts3_bsz512_lr8e_5",
    tokenized=mixture_config,
    model_config=llama_3_1_8b_instruct,
    sft_config=mixture_sft_config,
    tags=["llama", "openthoughts3", "sft"],
)


if __name__ == "__main__":
    executor_main(steps=[exp2199d_sft_llama_3_1_8b_instruct_openthoughts3])
