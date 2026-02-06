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
Fine-tunes Qwen/Qwen2.5-7B-Instruct on marin-community/open-thoughts-4-6k-math-qwen3-235b-a22b-neither-has-boxed.
This is a filtered subset where neither Qwen3-32B nor Qwen3-235B-A22B produced a boxed final answer,
using the Qwen3-235B-A22B reasoning traces. Part of teacher model comparison ablation study (GitHub #2262).
20 epochs.
"""
import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat
from experiments.chat_templates.qwen2pt5_instruct_chat_template import QWEN_2_5_INSTRUCT_CHAT_TEMPLATE
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
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import lm_mixture_data_config

SLUGIFY_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = SLUGIFY_PATTERN.sub("_", value.lower()).strip("_")
    return slug or "dataset"


def build_dataset_specs() -> tuple[dict[str, str], dict[str, int]]:
    datasets: dict[str, str] = {}
    weights: dict[str, int] = {}
    datasets["open-thoughts-4-6k-math-qwen3-235b-a22b-neither-has-boxed"] = "marin-community/open-thoughts-4-6k-math-qwen3-235b-a22b-neither-has-boxed"
    weights["open-thoughts-4-6k-math-qwen3-235b-a22b-neither-has-boxed"] = 6110
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
TARGET_EPOCHS = 20
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

mixture_sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-64"),
    tokenizer=qwen2_5_7b_instruct_tokenizer,
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=4e-5,
    max_seq_len=16384,
    seed=0,
    steps_per_checkpoint=(total_examples/TRAIN_BATCH_SIZE)//4,  # Every quarter epoch
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.999,
    pad_tokenizer_to_match_model=True,  # Pad tokenizer vocab to match model's vocab size (Qwen2.5-7B-Instruct model/tokenizer vocab sizes are different)
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

exp2262t_sft_qwen2pt5_7b_instruct_neither_boxed_235b = default_sft(
    name="exp2262t_ot4_neither_boxed_qwen3_235b_a22b_bsz128_lr4e_5",
    tokenized=mixture_config,
    model_config=qwen2_5_7b_instruct_16384_seq_len,
    sft_config=mixture_sft_config,
    tags=["qwen", "openthoughts4", "sft", "ablation", "neither-has-boxed"],
)

exp2262t_pt1_checkpoint = exp2262t_sft_qwen2pt5_7b_instruct_neither_boxed_235b.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    executor_main(steps=[exp2262t_sft_qwen2pt5_7b_instruct_neither_boxed_235b])
