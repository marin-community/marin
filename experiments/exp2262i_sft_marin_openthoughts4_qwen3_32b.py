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
Fine-tunes long context-extended Marin-8B model on marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens.
20 epochs.
"""
import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.tootsie.exp2062_long_context_8b import llama_8b_64k
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
    datasets["open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens"] = "marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens"
    weights["open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens"] = 29963
    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str) -> ExecutorStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
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

# Path to long-context Marin 8B checkpoint
longcontext_marin8b_checkpoint = InputName.hardcoded("checkpoints/tootsie-8b-giraffe-phase3-64k-21219c/hf/step-2999/")

mixture_sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-64"),
    tokenizer=marin_tokenizer,
    initialize_from_hf=longcontext_marin8b_checkpoint,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=1e-5,
    max_seq_len=16384,
    seed=0,
    steps_per_checkpoint=(total_examples/TRAIN_BATCH_SIZE)//4,  # Every quarter epoch
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.999,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=total_examples,  # IMPORTANT: Era shuffling (shuffle after every epoch). `shuffle=True` leads to same shuffle used in every epoch
    missing_weights_are_validation=True,
    mixture_block_size=12288,  # Doesn't matter for mixtures with 1 dataset
)

exp2262i_sft_marin_openthoughts4_qwen3_32b = default_sft(
    name="exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5",
    tokenized=mixture_config,
    model_config=llama_8b_64k,
    sft_config=mixture_sft_config,
    tags=["marin", "openthoughts4", "sft"],
)

exp2262i_pt1_checkpoint = exp2262i_sft_marin_openthoughts4_qwen3_32b.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    executor_main(steps=[exp2262i_sft_marin_openthoughts4_qwen3_32b])
