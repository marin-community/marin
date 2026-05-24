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
Fine-tunes Qwen3-8B (Qwen/Qwen3-8B) on the self-instilled rStarCoder dataset
(teetone/qwen3_8b_rstarcoder_instill_n8_valredundancy5_round1).

LR ablation of exp_sft_qwen3_8b_selfinstill_rstarcoder_n8_vr5_round1 — same
config except learning_rate raised from 5e-6 to 1e-5. Hypothesis: the original
LR was too conservative for rstarcoder's larger dataset (28850 vs code9k's
4739 examples), leading to insufficient per-step weight updates and a noisy
plateau in eval scores. The "lr1e5" suffix is in the name to force a distinct
executor hash so the new run gets a fresh GCS folder (no clobbering of the
original lr=5e-6 checkpoints).
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

EXPERIMENT_NAME = "exp_sft_qwen3_8b_selfinstill_lr1e5_rstarcoder_n8_vr5_round1"

# Dataset configuration
DATASET_ID = "teetone/qwen3_8b_rstarcoder_instill_n8_valredundancy5_round1"
DATASET_SHORT_NAME = "qwen3_8b_rstarcoder_n8_vr5_round1"
DATASET_SIZE = 28_850

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

RESOURCES = ResourceConfig.with_tpu("v5p-64", ram="400g")

mixture_sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=1e-5,  # raised from 5e-6 (the original 8B code9k recipe was too conservative for 28K examples)
    max_seq_len=32768,  # 32K context length
    seed=42,
    steps_per_checkpoint=20,  # frequent saves so preemptible v5p-32 bouncing doesn't wipe progress
    lr_schedule="cosine",
    warmup=0.05,
    decay=0.9,
    min_lr_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
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

exp_instilloracle_sft_qwen3_8b_selfinstill_lr1e5_rstarcoder_n8_vr5_round1 = default_sft(
    name=EXPERIMENT_NAME,
    tokenized=mixture_config,
    model_config=qwen3_8b_32k_tokens,
    sft_config=mixture_sft_config,
    tags=["qwen", "qwen3-8b", "rstarcoder", "code", "sft", "self-distillation", "lr1e5"],
)


if __name__ == "__main__":
    executor_main(steps=[exp_instilloracle_sft_qwen3_8b_selfinstill_lr1e5_rstarcoder_n8_vr5_round1])
