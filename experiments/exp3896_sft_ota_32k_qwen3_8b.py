# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce OpenThoughts-Agent's best 32K context SFT (laion/exp_tas_optimal_combined_traces).

Base model: Qwen/Qwen3-8B
Dataset: DCAgent/exp_tas_optimal_combined_traces (9,046 rows)
Hyperparams from model card: lr=4e-5, cosine + 10% warmup, 7 epochs, batch=16,
    AdamW (β=0.9/0.98), seed=42, 32K context.

Reported results: 23.8% TB-Lite, 12.6% TB2.

Tracked in: https://github.com/marin-community/marin/issues/3896

Usage:
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT marin \
        --env_vars TPU_CI true \
        --cluster us-east5-a \
        --no_wait \
        -- python experiments/exp3896_sft_ota_32k_qwen3_8b.py
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

# Dataset: DCAgent/exp_tas_optimal_combined_traces (9,046 rows)
DATASET_ID = "DCAgent/exp_tas_optimal_combined_traces"
NUM_SAMPLES = 9046


def build_dataset_specs() -> tuple[dict[str, str], dict[str, float]]:
    datasets: dict[str, str] = {}
    weights: dict[str, float] = {}
    datasets[DATASET_ID] = DATASET_ID
    weights[DATASET_ID] = float(NUM_SAMPLES)
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

# Hyperparams matching model card
TARGET_EPOCHS = 7
TRAIN_BATCH_SIZE = 16
MICROBATCH_SIZE = 16
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

# v5p-32 (16 chips) - batch=16 matches chip count exactly, same as exp2601
RESOURCES = ResourceConfig.with_tpu("v5p-32")

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=4e-5,  # Model card: 4e-5
    max_seq_len=32768,  # 32K context
    seed=42,
    steps_per_checkpoint=(total_examples // TRAIN_BATCH_SIZE) // 4,  # Every quarter epoch
    lr_schedule="cosine",  # Model card: cosine
    warmup=0.1,  # Model card: 10% warmup
    decay=0.9,
    weight_decay=0.0,
    beta1=0.9,  # Model card: 0.9
    beta2=0.98,  # Model card: 0.98
    epsilon=1e-8,
    pad_tokenizer_to_match_model=True,  # Qwen3 vocab size differs from tokenizer
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=total_examples,  # Era-based shuffling
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

qwen3_8b_32k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=32768,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

exp3896_sft_ota_32k = default_sft(
    name=f"exp3896_sft_ota_32k_qwen3_8b_32768tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_32k,
    sft_config=sft_config,
    tags=["qwen", "openthoughts-agent", "sft", "ota-32k", RESOURCE_SUFFIX],
)

exp3896_checkpoint = exp3896_sft_ota_32k.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3896: OT-Agent 32K SFT Reproduction ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}")
    executor_main(steps=[exp3896_sft_ota_32k])
