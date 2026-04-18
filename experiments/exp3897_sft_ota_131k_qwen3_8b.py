# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce OpenThoughts-Agent's best 131K context SFT (laion/GLM-4_7-r2egym_sandboxes-maxeps-131k-lc).

Base model: Qwen/Qwen3-8B (despite "GLM-4_7" in the model name, base is Qwen3-8B)
Dataset: DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k (4,521 rows)
Hyperparams from model card: lr=4e-5, cosine + 10% warmup, 7 epochs, batch=16,
    AdamW (beta=0.9/0.98), seed=42.

v2 fixes:
    - Fix 1: max_grad_norm=1e-4 (matching OT-Agent; was 1.0 Levanter default)
    - Fix 2: v5p-32 with batch=16 (matching OT-Agent's 16 GPUs; was v5p-256 batch=128)
    - Fix 3: replacements={} on dataset adapter to preserve native <think> tokens
    - Fix 4: YaRN RoPE scaling factor=4 from 32K base (matching OT-Agent's FP8+YARN config)

OOM Prevention Strategy (v5p-32 at 131K):
    - ScanCheckpointPolicy(save_carries="offload"): offload layer carries to host
    - per_device_parallelism=1: 1 example per chip
    - Splash attention (default on TPU)
    - ram="256g": XLA compilation needs >128 GB host RAM

Tracked in: https://github.com/marin-community/marin/issues/3897

Usage (Iris):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-32 \
        --memory 256GB \
        --region us-central1 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e MARIN_PREFIX gs://marin-us-central1 \
        --no-wait \
        -- python experiments/exp3897_sft_ota_131k_qwen3_8b.py
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

# Dataset: DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k (4,521 rows)
DATASET_ID = "DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k"
NUM_SAMPLES = 4521

# v5p-32 (16 chips) with batch=16: matches OT-Agent's 16 GPU setup.
# Override TPU via TPU_VARIANT env var (e.g. v5p-256); batch scales to 1 per chip.
# 256 GB host RAM: XLA compilation of the 131K graph with host offloading
# exceeds the default 128 GB.
TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-32")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, ram="256g")
NUM_CHIPS = RESOURCES.chip_count()
TRAIN_BATCH_SIZE = NUM_CHIPS  # 1 example per chip
MICROBATCH_SIZE = NUM_CHIPS


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
    max_grad_norm=1e-4,  # OT-Agent uses 1e-4 (acts as strong regularizer on small datasets)
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

# 131K context model config.
# YaRN RoPE scaling: extends 32K base context to 131K (factor=4.0), matching
# OT-Agent's FP8+YARN config (sft/llamafactory/examples/extras/fp8/).
# Carries-only offload: proven pattern from exp1295_32b/exp1395_qwen3_32b.
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

exp3897_sft_ota_131k_v2 = default_sft(
    name=f"exp3897v2_sft_ota_131k_qwen3_8b_131072tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_131k,
    sft_config=sft_config,
    tags=["qwen", "openthoughts-agent", "sft", "ota-131k", "long-context", "v2", RESOURCE_SUFFIX],
)

exp3897_checkpoint = exp3897_sft_ota_131k_v2.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3897v2: OT-Agent 131K SFT Reproduction (v2) ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}")
    print("Context length: 131,072")
    print("RoPE: YaRN factor=4.0 from 32K base")
    print(f"max_grad_norm: {sft_config.max_grad_norm}")
    executor_main(steps=[exp3897_sft_ota_131k_v2])
