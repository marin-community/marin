# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
131K SFT v2b: v5p-32 stability run without host offload or frequent temp checkpoints.

This variant exists to validate that 131K SFT can run stably on `v5p-32`
without the offload-related checkpoint OOMs seen in `v2`.

Design:
    - `v5p-32`, batch=16, 1 example per chip
    - `gradient_checkpointing=True` (no host offload)
    - 256 GB host RAM
    - time-based temporary checkpoints disabled
    - interim HF exports disabled
    - default run length is half of the full 7-epoch schedule

This isolates the suspected failure mode from the prior run:
Marin's default 10-minute temporary checkpoint cadence plus offloaded state
serialization was repeatedly OOM-killing the host during checkpoint commits.

Tracked in: https://github.com/marin-community/marin/issues/3897

Usage (default half-run stability test):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-32 \
        --memory 256GB \
        --region us-central1 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e MARIN_PREFIX gs://marin-us-central1 \
        --no-wait \
        -- python experiments/exp3897v2b_sft_ota_131k_qwen3_8b.py

Optional overrides:
    - `NUM_TRAIN_STEPS=<n>` to change the stability run length.
"""

import dataclasses
import math
import os

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

RESOURCES = ResourceConfig.with_tpu("v5p-32", ram="256g")
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
FULL_NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)
HALF_RUN_STEPS = (FULL_NUM_TRAIN_STEPS + 1) // 2
NUM_TRAIN_STEPS = int(os.environ.get("NUM_TRAIN_STEPS", str(HALF_RUN_STEPS)))

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
    steps_per_checkpoint=NUM_TRAIN_STEPS,
    steps_per_hf_export=-1,
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

qwen3_8b_131k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=131072,
    rope=YarnRotaryEmbeddingsConfig(
        theta=1_000_000.0,
        factor=4.0,
        original_max_position_embeddings=32768,
    ),
    gradient_checkpointing=True,
)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

base_step = default_sft(
    name=f"exp3897v2b_sft_ota_131k_qwen3_8b_131072tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_131k,
    sft_config=sft_config,
    tags=[
        "qwen",
        "openthoughts-agent",
        "sft",
        "ota-131k",
        "long-context",
        "v2b",
        "v5p32",
        "no-offload",
        "no-temp-checkpoints",
        RESOURCE_SUFFIX,
    ],
)

# Disable 10-minute temporary checkpoints for this stability run. The prior
# offload-based v5p-32 job OOM-killed in checkpoint serialization rather than
# during train steps, so we only keep the single permanent checkpoint at the end.
exp3897_sft_ota_131k_v2b = dataclasses.replace(
    base_step,
    config=dataclasses.replace(
        base_step.config,
        train_config=dataclasses.replace(
            base_step.config.train_config,
            trainer=dataclasses.replace(
                base_step.config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    base_step.config.train_config.trainer.checkpointer,
                    save_interval=None,
                ),
            ),
        ),
    ),
)

exp3897v2b_checkpoint = exp3897_sft_ota_131k_v2b.cd(f"checkpoints/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3897v2b: OT-Agent 131K SFT stability run on v5p-32 ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Full-train steps: {FULL_NUM_TRAIN_STEPS:,}")
    print(f"This run steps: {NUM_TRAIN_STEPS:,}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}, RAM: {RESOURCES.ram}")
    print("Gradient checkpointing: True (no offload)")
    print("Time-based temporary checkpoints: disabled")
    print("Interim HF exports: disabled")
    executor_main(steps=[exp3897_sft_ota_131k_v2b])
