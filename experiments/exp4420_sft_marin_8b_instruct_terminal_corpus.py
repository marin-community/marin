# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SFT of Marin-8B Instruct on Nemotron-Terminal-Corpus.

Trains on the full Terminal-Corpus (~366K examples across all 4 subsets) on v5p-64,
matching the paper hyperparams from the NemotronTerminal-8B reproduction (exp3490b).

Paper hyperparams: lr=2e-5, epochs=2, batch=128, seq_len=32768,
    AdamW (beta=0.9/0.95), cosine schedule, 10% warmup, grad_clip=1.0, wd=1e-4.

Tracked in: https://github.com/marin-community/marin/issues/4420

Usage (full corpus, default):
    uv run lib/marin/src/marin/run/ray_run.py \
        --env_vars WANDB_ENTITY marin-community \
        --env_vars WANDB_PROJECT marin \
        --env_vars TPU_CI true \
        --cluster us-east5-a \
        --no_wait \
        -- python experiments/exp4420_sft_marin_8b_instruct_terminal_corpus.py

Environment variables:
    - USE_FULL_CORPUS: use all 4 subsets (default: true). Set to false for skill-based only.
    - SYNTHETIC_DATA_FRACTION: fraction of skill-based data when USE_FULL_CORPUS=false (default: 0.05)
"""

import dataclasses
import math
import os

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.llama import llama_8b
from experiments.marin_models import MARIN_CHAT_TEMPLATE, marin_tokenizer
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# All subset sizes from HF dataset card
ALL_SUBSET_SIZES = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690,
}
SKILL_BASED_SUBSET_SIZES = {k: v for k, v in ALL_SUBSET_SIZES.items() if "skill_based" in k}

USE_FULL_CORPUS = os.environ.get("USE_FULL_CORPUS", "true").strip().lower() in {"true", "1", "yes"}

_TPU_VARIANT = os.environ.get("TPU_VARIANT", None)

if USE_FULL_CORPUS:
    SUBSET_SIZES = ALL_SUBSET_SIZES
    FRACTION = 1.0
    RESOURCES = ResourceConfig.with_tpu(_TPU_VARIANT or "v5p-64")
    _NUM_CHIPS = int(RESOURCES.device.variant.split("-")[-1]) // 2
    MICROBATCH_SIZE = min(128, _NUM_CHIPS)
else:
    SUBSET_SIZES = SKILL_BASED_SUBSET_SIZES
    FRACTION = float(os.environ.get("SYNTHETIC_DATA_FRACTION", "0.05"))
    RESOURCES = ResourceConfig.with_tpu(_TPU_VARIANT or "v5p-32")
    _NUM_CHIPS = int(RESOURCES.device.variant.split("-")[-1]) // 2
    MICROBATCH_SIZE = min(16, _NUM_CHIPS)

DATASETS = {k: k for k in SUBSET_SIZES}
WEIGHTS = {k: FRACTION * v for k, v in SUBSET_SIZES.items()}
EFFECTIVE_EXAMPLES = int(sum(WEIGHTS.values()))

# Training hyperparams (matching paper)
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
        name=f"{short_name.split('/')[-1]}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE),
    )


tokenized_datasets = {name: create_tokenization_step(dataset_id, name) for name, dataset_id in DATASETS.items()}

# Marin-8B Instruct uses the Llama 3 architecture. Llama3RotaryEmbeddingsConfig
# (theta=500000, factor=8, original_max_position_embeddings=8192) supports up to
# 131072 context natively, so 32768 is within range.
llama_8b_32k = dataclasses.replace(llama_8b, max_seq_len=32768)

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=marin_tokenizer,
    initialize_from_hf="marin-community/marin-8b-instruct",
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
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    WEIGHTS,
    shuffle=EFFECTIVE_EXAMPLES,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

if USE_FULL_CORPUS:
    corpus_tag = "full"
else:
    corpus_tag = f"{FRACTION:.0%}".replace("%", "pct")

exp4420_sft = default_sft(
    name=f"exp4420_sft_marin_8b_instruct_terminal_corpus_{corpus_tag}_32768tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=llama_8b_32k,
    sft_config=sft_config,
    tags=["llama", "marin-8b-instruct", "nemotron-terminal", "sft", "terminal-corpus", corpus_tag, RESOURCE_SUFFIX],
)

exp4420_checkpoint = exp4420_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp4420: Marin-8B Instruct Terminal-Corpus SFT ===")
    print(f"Mode: {'full corpus' if USE_FULL_CORPUS else f'{FRACTION:.0%} skill-based subset'}")
    print(f"Datasets: {len(SUBSET_SIZES)} subsets")
    print(f"Effective examples: {EFFECTIVE_EXAMPLES:,}")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}")
    executor_main(steps=[exp4420_sft])
