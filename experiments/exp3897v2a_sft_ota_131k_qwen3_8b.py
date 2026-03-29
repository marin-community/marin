# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
131K SFT v2a: sqrt-scaled LR and proportional grad_norm for batch=128.

Identical to v2 except LR and max_grad_norm are scaled for the 8x larger batch
(128 vs OT-Agent's 16):
    - learning_rate: 4e-5 * sqrt(8) = 1.13e-4
    - max_grad_norm: 1e-4 * 8 = 8e-4

v2 with batch=128 and max_grad_norm=1e-4 clipped every gradient by ~1300x,
causing the loss to plateau at 0.22 and SWE-bench to regress from 30% to 15%.

cf: https://github.com/marin-community/marin/issues/3897#issuecomment-4149562118

Tracked in: https://github.com/marin-community/marin/issues/3897

Usage (Iris):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-256 \
        --memory 256GB \
        --region us-central1 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e MARIN_PREFIX gs://marin-us-central1 \
        -e TPU_VARIANT v5p-256 \
        --no-wait \
        -- python experiments/exp3897v2a_sft_ota_131k_qwen3_8b.py
"""

import dataclasses
import math
import os

from haliax.nn.scan import ScanCheckpointPolicy
from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import YarnRotaryEmbeddingsConfig

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

DATASET_ID = "DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k"
NUM_SAMPLES = 4521

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-256")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, ram="256g")
NUM_CHIPS = RESOURCES.chip_count()
TRAIN_BATCH_SIZE = NUM_CHIPS  # 1 example per chip
MICROBATCH_SIZE = NUM_CHIPS

# Sqrt scaling for batch=128 vs OT-Agent's batch=16 (ratio=8):
#   lr: 4e-5 * sqrt(8) ≈ 1.13e-4
#   max_grad_norm: 1e-4 * 8 = 8e-4
BATCH_RATIO = TRAIN_BATCH_SIZE / 16
LEARNING_RATE = 4e-5 * math.sqrt(BATCH_RATIO)
MAX_GRAD_NORM = 1e-4 * BATCH_RATIO


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

TARGET_EPOCHS = 7
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=1,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    max_seq_len=131072,
    seed=42,
    steps_per_checkpoint=max(1, NUM_TRAIN_STEPS // 4),
    steps_per_hf_export=max(1, NUM_TRAIN_STEPS // 4),
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
    max_grad_norm=MAX_GRAD_NORM,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-8,
    pad_tokenizer_to_match_model=True,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=total_examples,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

qwen3_8b_131k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=131072,
    rope=YarnRotaryEmbeddingsConfig(
        theta=1_000_000.0,
        factor=4.0,
        original_max_position_embeddings=32768,
    ),
    gradient_checkpointing=ScanCheckpointPolicy(save_carries="offload"),
)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

exp3897_sft_ota_131k_v2a = default_sft(
    name=f"exp3897v2a_sft_ota_131k_qwen3_8b_131072tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_131k,
    sft_config=sft_config,
    tags=["qwen", "openthoughts-agent", "sft", "ota-131k", "long-context", "v2a", "sqrt-scaling", RESOURCE_SUFFIX],
)

exp3897v2a_checkpoint = exp3897_sft_ota_131k_v2a.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3897v2a: OT-Agent 131K SFT (sqrt scaling) ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (ratio={BATCH_RATIO:.0f}x vs OT-Agent)")
    print(f"LR: {LEARNING_RATE:.2e} (4e-5 * sqrt({BATCH_RATIO:.0f}))")
    print(f"max_grad_norm: {MAX_GRAD_NORM:.1e} (1e-4 * {BATCH_RATIO:.0f})")
    print(f"Resources: {RESOURCES.device.variant} ({NUM_CHIPS} chips)")
    executor_main(steps=[exp3897_sft_ota_131k_v2a])
