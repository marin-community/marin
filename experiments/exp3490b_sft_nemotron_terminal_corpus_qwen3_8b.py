# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SFT reproduction of Nemotron-Terminal-8B using Terminal-Corpus on Qwen3-8B.

By default, trains on the full Terminal-Corpus (~366K examples across all 4 subsets)
on v5p-64. Set USE_FULL_CORPUS=false for the smaller 5% skill-based subset on v5p-32.

Paper hyperparams: lr=2e-5, epochs=2, batch=128, seq_len=32768,
    AdamW (β=0.9/0.95), cosine schedule, 10% warmup, grad_clip=1.0, wd=1e-4.
Paper reports 13.0% on TB2 for the full run (with 490K examples including unreleased
seed-based tasks; we use the 366K available on HF).

Tracked in: https://github.com/marin-community/marin/issues/3490

Usage (full corpus, default):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT marin \
        --env_vars TPU_CI true \
        --cluster us-east5-a \
        --no_wait \
        -- python experiments/exp3490b_sft_nemotron_terminal_corpus_qwen3_8b.py

Usage (5% skill-based subset):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT marin \
        --env_vars TPU_CI true \
        --env_vars USE_FULL_CORPUS false \
        --cluster us-east5-a \
        --no_wait \
        -- python experiments/exp3490b_sft_nemotron_terminal_corpus_qwen3_8b.py

Environment variables:
    - USE_FULL_CORPUS: use all 4 subsets (default: true). Set to false for skill-based only.
    - SYNTHETIC_DATA_FRACTION: fraction of skill-based data when USE_FULL_CORPUS=false (default: 0.05)
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

import os

# All subset sizes from HF dataset card
ALL_SUBSET_SIZES = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690,
}
SKILL_BASED_SUBSET_SIZES = {k: v for k, v in ALL_SUBSET_SIZES.items() if "skill_based" in k}

USE_FULL_CORPUS = os.environ.get("USE_FULL_CORPUS", "true").strip().lower() in {"true", "1", "yes"}

if USE_FULL_CORPUS:
    SUBSET_SIZES = ALL_SUBSET_SIZES
    FRACTION = 1.0
    RESOURCES = ResourceConfig.with_tpu("v5p-64")
    MICROBATCH_SIZE = 32
else:
    SUBSET_SIZES = SKILL_BASED_SUBSET_SIZES
    FRACTION = float(os.environ.get("SYNTHETIC_DATA_FRACTION", "0.05"))
    RESOURCES = ResourceConfig.with_tpu("v5p-32")
    MICROBATCH_SIZE = 16

DATASETS = {k: k for k in SUBSET_SIZES}
WEIGHTS = {k: FRACTION * v for k, v in SUBSET_SIZES.items()}
EFFECTIVE_EXAMPLES = int(sum(WEIGHTS.values()))

# Training hyperparams (matching paper where possible)
TARGET_EPOCHS = 2
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * EFFECTIVE_EXAMPLES / TRAIN_BATCH_SIZE))

WARMUP_FRACTION = 0.1
DECAY_FRACTION = 1.0 - WARMUP_FRACTION

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"


def create_tokenization_step(dataset_identifier: str, short_name: str):
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name.split('/')[-1]}_qwen3_8b_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen3_8b_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
    )


tokenized_datasets = {name: create_tokenization_step(dataset_id, name) for name, dataset_id in DATASETS.items()}

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=2e-5,  # Paper: 2e-5
    max_seq_len=32768,  # Paper: 32768
    seed=42,
    steps_per_checkpoint=max(1, NUM_TRAIN_STEPS // 4),
    lr_schedule="cosine",  # Paper: cosine
    warmup=WARMUP_FRACTION,  # Paper: 10% warmup
    decay=DECAY_FRACTION,
    weight_decay=1e-4,  # Paper: 1e-4
    beta1=0.9,  # Paper: 0.9
    beta2=0.95,  # Paper: 0.95
    max_grad_norm=1.0,  # Paper: 1.0
    pad_tokenizer_to_match_model=True,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    WEIGHTS,
    shuffle=EFFECTIVE_EXAMPLES,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

qwen3_8b_32k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=32768,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
)

if USE_FULL_CORPUS:
    corpus_tag = "full"
else:
    corpus_tag = f"{FRACTION:.0%}".replace("%", "pct")

exp3490b_sft = default_sft(
    name=f"exp3490b_sft_nemotron_terminal_corpus_{corpus_tag}_qwen3_8b_32768tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_32k,
    sft_config=sft_config,
    tags=["qwen", "nemotron-terminal", "sft", "terminal-corpus", corpus_tag, RESOURCE_SUFFIX],
)

exp3490b_checkpoint = exp3490b_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp3490b: Nemotron-Terminal-Corpus SFT ===")
    print(f"Mode: {'full corpus' if USE_FULL_CORPUS else f'{FRACTION:.0%} skill-based subset'}")
    print(f"Datasets: {len(SUBSET_SIZES)} subsets")
    print(f"Effective examples: {EFFECTIVE_EXAMPLES:,}")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}")
    executor_main(steps=[exp3490b_sft])
