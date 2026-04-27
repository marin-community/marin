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
Fine-tunes Qwen3-8B (Qwen/Qwen3-8B) on the self-instilled OpenThoughts4 Code dataset
(teetone/qwen3_8b_openthoughts4_code9K_instill_n8_valredundancy5_round1).

Hyperparameter variant of the round1 experiment:
- Lower LR (5e-6 vs 1e-5) to reduce oscillation
- Fewer steps (2000 vs 4000) to avoid overfitting
- Weight decay 0.01 for regularization
- Relaxed grad clipping (1.0 vs 0.2)
- Slightly longer warmup (0.05 vs 0.03)
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
from marin.processing.tokenize import lm_mixture_data_config

EXPERIMENT_NAME = "exp_sft_qwen3_8b_selfinstill_ot4_code9k_n8_vr5_2k_lr5e6_wd01"

# Dataset configuration
DATASET_ID = "teetone/qwen3_8b_openthoughts4_code9K_instill_n8_valredundancy5_round1"
DATASET_SHORT_NAME = "qwen3_8b_ot4_code9k_n8_vr5_round1"
DATASET_SIZE = 4_739

dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[DATASET_ID]
dataset = get_instruction_dataset(DATASET_ID, splits=dataset_config.splits)

tokenized_selfinstill_code = default_tokenize(
    name=f"{DATASET_SHORT_NAME}_qwen3_8b_tokenizer",
    dataset=dataset / "**/*.jsonl.gz",
    tokenizer=qwen3_8b_tokenizer,
    format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
)

tokenized_datasets = {DATASET_SHORT_NAME: tokenized_selfinstill_code}
mixture_weights = {DATASET_SHORT_NAME: DATASET_SIZE}

# Training configuration
TARGET_EPOCHS = 8
TRAIN_BATCH_SIZE = 64
MICROBATCH_SIZE = 32  # 2 gradient accumulation steps (v4-64 OOMs at 64)

# Fix at 4000 instead of using the number of epochs
# NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * DATASET_SIZE / TRAIN_BATCH_SIZE)
NUM_TRAIN_STEPS = 2000

RESOURCES = ResourceConfig.with_tpu("v5p-64")

mixture_sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-6,
    max_seq_len=32768,  # 32K context length
    seed=42,
    steps_per_checkpoint=(DATASET_SIZE / TRAIN_BATCH_SIZE) // 4,  # Every quarter epoch
    lr_schedule="cosine",
    warmup=0.05,  # Slightly longer warmup for stability with lower LR
    decay=0.9,
    min_lr_ratio=0.1,  # min_lr_rate from Open-R1 recipe (cosine_with_min_lr)
    weight_decay=0.01,  # Light regularization to prevent overfitting
    max_grad_norm=1.0,  # Standard value; 0.2 was too aggressive
    beta1=0.9,
    beta2=0.999,
    pad_tokenizer_to_match_model=True,
    steps_per_hf_export=100,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=DATASET_SIZE,  # Era shuffling
    missing_weights_are_validation=True,
    mixture_block_size=32768,  # Match max_seq_len
)

# Model config with 32K context length
qwen3_8b_32k_tokens = dataclasses.replace(
    qwen3_8b,
    max_seq_len=32768,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
)

exp_instilloracle_sft_qwen3_8b_selfinstill_ot4_code9k_n8_vr5_round1_lr5e6_wd01_2ksteps = default_sft(
    name=EXPERIMENT_NAME,
    tokenized=mixture_config,
    model_config=qwen3_8b_32k_tokens,
    sft_config=mixture_sft_config,
    tags=["qwen", "qwen3-8b", "openthoughts4", "code", "sft", "self-distillation"],
)


if __name__ == "__main__":
    executor_main(steps=[exp_instilloracle_sft_qwen3_8b_selfinstill_ot4_code9k_n8_vr5_round1_lr5e6_wd01_2ksteps])
