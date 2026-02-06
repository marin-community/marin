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
Resumes experiments/exp2199b_redo_sft_qwen2pt5_7b_instruct_openthoughts3.py and runs more epochs.
"""
import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat
from experiments.qwen2pt5_instruct_chat_template import QWEN_2_5_INSTRUCT_CHAT_TEMPLATE

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.exp2199b_redo_sft_qwen2pt5_7b_instruct_openthoughts3 import exp2199b_sft_qwen2pt5_7b_instruct_openthoughts3, mixture_config
from experiments.qwen3 import qwen2_5_7b_instruct, qwen2_5_7b_instruct_tokenizer
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, InputName, versioned
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
        name=f"{short_name}_qwen2_5_7b_instruct_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen2_5_7b_instruct_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_2_5_INSTRUCT_CHAT_TEMPLATE),
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = sum(mixture_weights.values())
TARGET_EPOCHS = 10  # Changed from 5
TRAIN_BATCH_SIZE = 512
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

# Part 2: Resume from last checkpoint of previous run
part1_final_checkpoint = InputName.hardcoded("checkpoints/exp2199b_redo_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-2d659d/checkpoints/step-11700/")
pt2_train_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v4-512"),
    tokenizer=qwen2_5_7b_instruct_tokenizer,
    initialize_from_checkpoint_path=part1_final_checkpoint,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=versioned(8e-5),
    max_seq_len=16384,
    seed=0,
    steps_per_checkpoint=(total_examples/TRAIN_BATCH_SIZE)//4,  # Every quarter epoch
    lr_schedule="cosine",
    warmup=versioned(0.1),
    decay=versioned(0.9),
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.999,
)
exp2199b_redo_pt2_sft_qwen2pt5_7b_instruct_openthoughts3_pt2 = default_sft(
    name="exp2199b_redo_pt2_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5",
    tokenized=mixture_config,
    model_config=qwen2_5_7b_instruct,
    sft_config=pt2_train_config,
    tags=["qwen", "openthoughts3", "sft"],
)


if __name__ == "__main__":
    executor_main(steps=[exp2199b_redo_pt2_sft_qwen2pt5_7b_instruct_openthoughts3_pt2])
