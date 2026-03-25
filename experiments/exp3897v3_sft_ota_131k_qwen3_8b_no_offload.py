# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
131K SFT without gradient offloading (v3).

Same as v2 (exp3897_sft_ota_131k_qwen3_8b.py) but uses default
gradient_checkpointing=True instead of carries-only offload. The offload
adds host<->device transfers every layer which is slow (~218s/step).
Without offloading, activations stay on-chip — should be significantly faster.

Per-chip HBM estimate: ~3 GB carries + ~1.5 GB model/optimizer = ~5 GB per chip
out of 95 GB available on v5p-32 (16 chips). Fits comfortably.

Host RAM set to 512 GB to ensure XLA compilation succeeds (the offload graph
needed >128 GB; the simpler non-offload graph likely needs less, but we start
generous and can reduce later).

Tracked in: https://github.com/marin-community/marin/issues/3897

Usage (Iris):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-32 \
        --memory 512GB \
        --region us-central1 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e MARIN_PREFIX gs://marin-us-central1 \
        --no-wait \
        -- python experiments/exp3897v3_sft_ota_131k_qwen3_8b_no_offload.py
"""

import dataclasses
import math

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

# Dataset: DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k (4,521 rows)
DATASET_ID = "DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k"
NUM_SAMPLES = 4521

# v5p-32 (16 chips) with batch=16.
# 512 GB host RAM: generous allocation for XLA compilation of 131K graph
# without offloading. Can reduce if compilation fits in less.
RESOURCES = ResourceConfig.with_tpu("v5p-32", ram="512g")
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
    learning_rate=4e-5,
    max_seq_len=131072,
    seed=42,
    steps_per_checkpoint=max(1, NUM_TRAIN_STEPS // 4),
    steps_per_hf_export=max(1, NUM_TRAIN_STEPS // 4),
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
    max_grad_norm=1e-4,
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

# 131K context, YaRN RoPE, DEFAULT gradient checkpointing (no offload).
# gradient_checkpointing=True saves inter-layer carries on device and
# recomputes block internals — no host transfers needed.
qwen3_8b_131k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=131072,
    rope=YarnRotaryEmbeddingsConfig(
        theta=1_000_000.0,
        factor=4.0,
        original_max_position_embeddings=32768,
    ),
    # Default gradient_checkpointing=True — no offloading, all on-chip.
)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

exp3897_sft_ota_131k_v3 = default_sft(
    name=f"exp3897v3_sft_ota_131k_qwen3_8b_131072tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_131k,
    sft_config=sft_config,
    tags=["qwen", "openthoughts-agent", "sft", "ota-131k", "long-context", "v3", "no-offload", RESOURCE_SUFFIX],
)

exp3897v3_checkpoint = exp3897_sft_ota_131k_v3.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3897v3: OT-Agent 131K SFT (no offload) ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}, RAM: {RESOURCES.ram}")
    print(f"Gradient checkpointing: {qwen3_8b_131k.gradient_checkpointing} (no offload)")
    executor_main(steps=[exp3897_sft_ota_131k_v3])
