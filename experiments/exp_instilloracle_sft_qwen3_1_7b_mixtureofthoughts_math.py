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
Fine-tunes Qwen3-1.7B (Qwen/Qwen3-1.7B) on the Mixture-of-Thoughts Math dataset
(open-r1/Mixture-of-Thoughts, math subset).

This experiment is adapted from the Open-R1 distillation recipe.
"""
import dataclasses
import math

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.qwen3 import qwen3_1_7b, qwen3_1_7b_tokenizer
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# Dataset configuration
DATASET_ID = "open-r1/Mixture-of-Thoughts-Math"
DATASET_SHORT_NAME = "mixtureofthoughts_math"
DATASET_SIZE = 93_734

dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[DATASET_ID]
dataset = get_instruction_dataset(DATASET_ID, splits=dataset_config.splits)

tokenized_mot_math = default_tokenize(
    name=f"{DATASET_SHORT_NAME}_qwen3_1_7b_tokenizer",
    dataset=dataset / "**/*.jsonl.gz",
    tokenizer=qwen3_1_7b_tokenizer,
    format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
)

tokenized_datasets = {DATASET_SHORT_NAME: tokenized_mot_math}
mixture_weights = {DATASET_SHORT_NAME: DATASET_SIZE}

# Training configuration
# Using v5p-16 (8 chips)
TARGET_EPOCHS = 8
TRAIN_BATCH_SIZE = 128  # 16 per device on v5p-16 (8 chips)
MICROBATCH_SIZE = 128  # No gradient accumulation
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * DATASET_SIZE / TRAIN_BATCH_SIZE)

RESOURCES = ResourceConfig.with_tpu("v5p-16")

mixture_sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_1_7b_tokenizer,
    model_name_or_path="Qwen/Qwen3-1.7B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-5,  # Higher LR for smaller model (vs 3e-5 for 4B)
    max_seq_len=32768,  # 32K context length
    seed=42,
    steps_per_checkpoint=int((DATASET_SIZE / TRAIN_BATCH_SIZE) // 4),  # Every quarter epoch
    lr_schedule="cosine",
    warmup=0.03,  # warmup_ratio (same ratio, fewer absolute steps)
    decay=0.9,
    min_lr_ratio=0.1,  # min_lr_rate from Open-R1 recipe (cosine_with_min_lr)
    weight_decay=0.0,
    max_grad_norm=0.2,  # From Open-R1 recipe
    beta1=0.9,
    beta2=0.999,
    pad_tokenizer_to_match_model=True,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    permutation_type="feistel",
    shuffle=DATASET_SIZE,  # Era shuffling
    missing_weights_are_validation=True,
    mixture_block_size=32768,  # Match max_seq_len
)

# Model config - qwen3_1_7b already has 32K context and rope_theta=1M
# Only add cross_entropy_block_size for memory optimization
qwen3_1_7b_sft = dataclasses.replace(
    qwen3_1_7b,
    cross_entropy_block_size=32000,  # Process vocab in chunks to reduce memory during loss computation
)

exp_instilloracle_sft_qwen3_1_7b_mot_math = default_sft(
    name="exp_instilloracle_sft_qwen3_1_7b_mixtureofthoughts_math",
    tokenized=mixture_config,
    model_config=qwen3_1_7b_sft,
    sft_config=mixture_sft_config,
    tags=["qwen", "qwen3-1.7b", "mixture-of-thoughts", "math", "sft", "distillation"],
)


if __name__ == "__main__":
    executor_main(steps=[exp_instilloracle_sft_qwen3_1_7b_mot_math])
