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
Debug config to sanity check that we can fine-tune Qwen2.5-32B-Instruct on a dataset without errors.
"""
import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat
from experiments.qwen2pt5_instruct_chat_template import QWEN_2_5_INSTRUCT_CHAT_TEMPLATE

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.qwen3 import qwen2_5_32b_instruct, qwen2_5_32b_instruct_tokenizer
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
    datasets["openthoughts4_1pt2m_qwen3_3b"] = "marin-community/OpenThoughts4-1.2M-Qwen3-32B"
    weights["openthoughts4_1pt2m_qwen3_3b"] = 1200000  # Has exactly 1.2M rows
    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str) -> ExecutorStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name}_qwen2_5_32b_instruct_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen2_5_32b_instruct_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_2_5_INSTRUCT_CHAT_TEMPLATE),
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = sum(mixture_weights.values())
TARGET_EPOCHS = 5
TRAIN_BATCH_SIZE = 256
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

mixture_sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-64", slice_count=2),
    tokenizer=qwen2_5_32b_instruct_tokenizer,
    model_name_or_path="Qwen/Qwen2.5-32B-Instruct",
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=8e-5,
    max_seq_len=16384,
    seed=0,
    steps_per_checkpoint=(total_examples/TRAIN_BATCH_SIZE)//4,  # Every quarter epoch
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

exp2209d2_sft_qwen2pt5_32b_instruct_openthoughts4_1pt2m_qwen3_3b = default_sft(
    name="exp2209debug2_sft_qwen2pt5_32b_instruct_openthoughts4_1pt2m_qwen3_3b_bsz512_lr8e_5",
    tokenized=mixture_config,
    model_config=qwen2_5_32b_instruct,
    sft_config=mixture_sft_config,
    tags=["qwen", "openthoughts4_1pt2m_qwen3_3b", "sft"],
)


if __name__ == "__main__":
    executor_main(steps=[exp2209d2_sft_qwen2pt5_32b_instruct_openthoughts4_1pt2m_qwen3_3b])
