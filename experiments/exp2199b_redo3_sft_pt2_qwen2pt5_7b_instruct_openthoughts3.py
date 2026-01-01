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
Fine-tunes Qwen2.5-7B-Instruct (Qwen/Qwen2.5-7B-Instruct) on the OpenThoughts3 dataset (open-thoughts/OpenThoughts3-1.2M).
REDO #1: After fixing custom chat template (QWEN_2_5_INSTRUCT_CHAT_TEMPLATE) and changing beta2 to 0.999.
REDO #2:
  - Reduced LR (8e-5 -> 4e-5) to reduce loss spikes
  - Increased batch size (512 -> 1024)
  - Extending to 20 epochs with diff LR schedule since our train loss decreases slower than what OT3 folks saw.
REDO #3:
  - Includes changes from redo #1 (fixed custom Qwen2.5 chat template, updated beta2=0.999)
  - Does NOT include optimization-related changes from redo #2 (changes to learning rate, batch size, # epoch)
  - Does include fixed shuffling from redo #2 (era shuffling to shuffle dataset every epoch)
  - Fixes max_seq_len bug: actually trains ons 16384 seq len (instead of 4096 as before)
"""
import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.qwen3 import qwen2_5_7b_instruct, qwen2_5_7b_instruct_tokenizer
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, InputName
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
        format=ChatLmDatasetFormat(),
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = sum(mixture_weights.values())
TARGET_EPOCHS = 5
TRAIN_BATCH_SIZE = 512
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

# Part 2: Resume from last checkpoint of previous run
part1_final_checkpoint = InputName.hardcoded("checkpoints/exp2199b_redo3_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-c05011/hf/step-11718/")

mixture_sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v4-512"),
    tokenizer=qwen2_5_7b_instruct_tokenizer,
    initialize_from_hf=part1_final_checkpoint,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=1e-5,
    max_seq_len=16384,
    seed=0,
    steps_per_checkpoint=(total_examples/TRAIN_BATCH_SIZE)//4,  # Every quarter epoch
    lr_schedule="cosine",
    warmup=0.05,
    decay=0.95,
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.999,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    permutation_type="feistel",
    shuffle=total_examples,  # IMPORTANT: Era shuffling (shuffle after every epoch). `shuffle=True` leads to same shuffle used in every epoch
    missing_weights_are_validation=True,
    mixture_block_size=12288,  # Doesn't matter for mixtures with 1 dataset
)

qwen2_5_7b_instruct_16384_seq_len = dataclasses.replace(
    qwen2_5_7b_instruct,
    max_seq_len=16384,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
)

exp2199b_sft_qwen2pt5_7b_instruct_openthoughts3 = default_sft(
    name="exp2199b_redo3_sft_pt2_qwen2pt5_7b_instruct_ot3_bsz512_lr1e_5",
    tokenized=mixture_config,
    model_config=qwen2_5_7b_instruct_16384_seq_len,
    sft_config=mixture_sft_config,
    tags=["qwen", "openthoughts3", "sft"],
)


if __name__ == "__main__":
    executor_main(steps=[exp2199b_sft_qwen2pt5_7b_instruct_openthoughts3])
