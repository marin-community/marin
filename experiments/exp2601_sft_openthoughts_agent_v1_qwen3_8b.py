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
Fine-tunes Qwen/Qwen3-8B on OpenThoughts-Agent-v1-SFT (open-thoughts/OpenThoughts-Agent-v1-SFT).

OpenThoughts-Agent-v1-SFT is exported from Harbor traces. Each row contains a multi-turn "conversations"
column (Harbor SFT format) plus metadata like agent, model, task, etc.
"""

import dataclasses
import math

from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.qwen3 import qwen3_8b, qwen3_8b_tokenizer
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import TokenizerStep, lm_mixture_data_config


def build_dataset_specs() -> tuple[dict[str, str], dict[str, float]]:
    datasets: dict[str, str] = {}
    weights: dict[str, float] = {}

    ota_v1_sft_dataset = "open-thoughts/OpenThoughts-Agent-v1-SFT"
    ota_v1_sft_num_samples = 15209

    datasets[ota_v1_sft_dataset] = ota_v1_sft_dataset
    weights[ota_v1_sft_dataset] = float(ota_v1_sft_num_samples)

    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str) -> TokenizerStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name.split('/')[-1]}_qwen3_8b_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen3_8b_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = int(sum(mixture_weights.values()))
TARGET_EPOCHS = 7
TRAIN_BATCH_SIZE = 16
MICROBATCH_SIZE = 16
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

# Use v5p-32 (16 chips) - batch=16 matches chip count exactly, no TP needed
RESOURCES = ResourceConfig.with_tpu("v5p-32")
TENSOR_PARALLEL_SIZE = 1

mixture_sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    model_name_or_path="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(
        TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES, TENSOR_PARALLEL_SIZE
    ),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=4e-5,
    max_seq_len=32768,
    seed=42,
    steps_per_checkpoint=(total_examples // TRAIN_BATCH_SIZE) // 4,  # Every quarter epoch
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
    max_grad_norm=1e-4,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-8,
    pad_tokenizer_to_match_model=True,  # Model and tokenizer vocab sizes differ
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    permutation_type="feistel",
    # IMPORTANT: Era shuffling (shuffle after every epoch). `shuffle=True` repeats the same shuffle each epoch.
    shuffle=total_examples,
    missing_weights_are_validation=True,
    mixture_block_size=12288,  # Doesn't matter for mixtures with 1 dataset
)

qwen3_8b_32768_seq_len = dataclasses.replace(
    qwen3_8b,
    max_seq_len=32768,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
    cross_entropy_block_size=32000,  # Process vocab in chunks to reduce memory during loss computation
)

# Derive resource name for experiment suffix (e.g., "v5p-64" -> "v5p64")
RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

exp2601_sft_openthoughts_agent_v1_qwen3_8b = default_sft(
    name=f"exp2601_sft_ot_agent_v1_qwen3_8b_32768tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_32768_seq_len,
    sft_config=mixture_sft_config,
    tags=["qwen", "openthoughts-agent", "sft", RESOURCE_SUFFIX],
)

exp2601_checkpoint = exp2601_sft_openthoughts_agent_v1_qwen3_8b.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()


if __name__ == "__main__":
    executor_main(steps=[exp2601_sft_openthoughts_agent_v1_qwen3_8b])
