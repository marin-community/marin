# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce OpenThoughts-Agent's best 131K context SFT (laion/GLM-4_7-r2egym_sandboxes-maxeps-131k-lc).

Base model: Qwen/Qwen3-8B (despite "GLM-4_7" in the model name, base is Qwen3-8B)
Dataset: DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k (4,521 rows)
Hyperparams from model card: lr=4e-5, cosine + 10% warmup, 7 epochs, batch=16,
    AdamW (β=0.9/0.98), seed=42.

OOM Prevention Strategy:
    131K context on Qwen3-8B requires aggressive memory optimization.
    - gradient_checkpointing="offload": offload layer carries to host memory
    - per_device_parallelism=1: minimize per-chip activation memory
    - Splash attention (default on TPU): O(seq*d) not O(seq^2) memory
    - per_device_eval_parallelism=1: avoid OOM during eval
    If OOM persists, escalate to gradient_checkpointing="recompute".

Tracked in: https://github.com/marin-community/marin/issues/3897

Usage (Iris, preferred for stability):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-64 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        --no-wait \
        -- python experiments/exp3897_sft_ota_131k_qwen3_8b.py

Usage (Ray, fallback):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY=marin-community \
        --env_vars WANDB_PROJECT=marin \
        --env_vars TPU_CI=true \
        --cluster us-east5-a \
        --no_wait \
        -- python experiments/exp3897_sft_ota_131k_qwen3_8b.py
"""

import dataclasses
import math
import os

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

# Dataset: DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k (4,521 rows)
DATASET_ID = "DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k"
NUM_SAMPLES = 4521

# --- TPU and batch configuration ---
# v5p-64 (32 chips, 95 GB HBM each = 3.04 TB total): preferred for 131K context.
# batch=32 is the minimum for v5p-64 (1 example per chip, no accumulation).
# The model card used batch=16 on 16 GPUs; we use batch=32 on 32 chips.
# If v5p-64 unavailable, fall back to v5p-32 via TPU_VARIANT env var.
TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-64")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT)
NUM_CHIPS = RESOURCES.chip_count()

if NUM_CHIPS >= 32:
    TRAIN_BATCH_SIZE = 32
    MICROBATCH_SIZE = 32
else:
    TRAIN_BATCH_SIZE = 16
    MICROBATCH_SIZE = 16


def build_dataset_specs() -> tuple[dict[str, str], dict[str, float]]:
    datasets: dict[str, str] = {}
    weights: dict[str, float] = {}
    datasets[DATASET_ID] = DATASET_ID
    weights[DATASET_ID] = float(NUM_SAMPLES)
    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str):
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
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=1,  # Minimize eval memory at 131K seq len
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=4e-5,  # Model card: 4e-5
    max_seq_len=131072,  # 131K context
    seed=42,
    steps_per_checkpoint=max(1, NUM_TRAIN_STEPS // 4),
    steps_per_hf_export=max(1, NUM_TRAIN_STEPS // 4),
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

# 131K context model config with aggressive gradient checkpointing.
# "offload" moves layer carries and inputs to host pinned memory between layers,
# drastically reducing HBM usage. Splash attention is automatic on TPU.
qwen3_8b_131k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=131072,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
    gradient_checkpointing="offload",
    # Use 256 block size for splash attention to avoid XLA sublane assertion failure
    # at 131K seq len (default 512 triggers async_dynamic_index_emitter.cc crash).
    flash_attention_block_size=256,
)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

exp3897_sft_ota_131k = default_sft(
    name=f"exp3897_sft_ota_131k_qwen3_8b_131072tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_131k,
    sft_config=sft_config,
    tags=["qwen", "openthoughts-agent", "sft", "ota-131k", "long-context", RESOURCE_SUFFIX],
)

exp3897_checkpoint = exp3897_sft_ota_131k.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3897: OT-Agent 131K SFT Reproduction ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Microbatch size: {MICROBATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant} ({NUM_CHIPS} chips)")
    print("Context length: 131,072")
    print("Gradient checkpointing: offload")
    print(f"Per-device parallelism: {sft_config.per_device_parallelism}")
    executor_main(steps=[exp3897_sft_ota_131k])
