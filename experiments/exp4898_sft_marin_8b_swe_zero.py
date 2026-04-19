# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Quality validation: Marin-8B base SFT on SWE-ZERO trajectories.

Continue-trains Marin-8B base on subsets of SWE-ZERO trajectories
(AlienKevin/SWE-ZERO-12M-trajectories) and evaluates on SWE-bench
Verified/Multilingual to validate trajectory quality.

Three subset sizes for scaling law: 10K, 50K, 100K trajectories.

Key design decisions:
- LR=1e-4 (5x higher than 2e-5 used in exp4420, which underperformed
  at 1.1% TB2 due to Marin-8B's weight decay during pretraining; see #4225)
- 1 epoch (speed — quality signal detectable in 1 epoch)
- 32K context (matches SWE-ZERO generation config)
- mini-swe-agent v1 chat format (role/content, bash_command as metadata)

Tracked in: https://github.com/marin-community/marin/issues/4898

Usage:
    # 10K subset (fastest, ~6h on v5p-32)
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-32 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e SWE_ZERO_SUBSET 10000 \
        --no-wait \
        -- python experiments/exp4898_sft_marin_8b_swe_zero.py

    # 50K subset (~30h on v5p-32)
    SWE_ZERO_SUBSET=50000

    # 100K subset (~60h on v5p-32, or ~30h on v5p-64)
    SWE_ZERO_SUBSET=100000
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
    InstructionDatasetConfig,
    get_instruction_dataset,
    multi_turn_adapter,
)
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# Dataset: SWE-ZERO trajectories (mini-swe-agent v1 format)
DATASET_ID = "AlienKevin/SWE-ZERO-12M-trajectories"
TOTAL_AVAILABLE = 1_453_008  # 20B checkpoint

# Subset size from env (default 10K for fast iteration)
SUBSET_SIZE = int(os.environ.get("SWE_ZERO_SUBSET", "10000"))

# TPU variant from env (default v5p-32)
TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-32")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT)
_NUM_CHIPS = int(RESOURCES.device.variant.split("-")[-1]) // 2

# Training hyperparams
TARGET_EPOCHS = 1
TRAIN_BATCH_SIZE = min(16, _NUM_CHIPS)
MICROBATCH_SIZE = TRAIN_BATCH_SIZE
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * SUBSET_SIZE / TRAIN_BATCH_SIZE))

# LR=1e-4: higher than the 2e-5 that failed for Marin-8B in exp4420
# (see #4225 for weight decay / LR interaction analysis)
LEARNING_RATE = 1e-4

# Register the SWE-ZERO dataset in the global config map.
# Messages are in chat format: [{role, content, bash_command}].
# multi_turn_adapter reads role+content, ignores bash_command (metadata).
INSTRUCTION_DATASET_NAME_TO_CONFIG[DATASET_ID] = InstructionDatasetConfig(
    hf_dataset_id=DATASET_ID,
    revision="main",
    adapter=multi_turn_adapter(conversation_column="messages"),
    metadata_columns=["instance_id", "repo", "exit_status"],
    splits=["train"],
)


def create_tokenization_step():
    dataset = get_instruction_dataset(DATASET_ID, splits=["train"])
    return default_tokenize(
        name=f"swe_zero_{SUBSET_SIZE // 1000}k_marin_tokenizer_v3",  # v2: cache-bust after dataset cleanup
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE),
    )


tokenized = create_tokenization_step()

mixture_config = lm_mixture_data_config(
    {DATASET_ID: tokenized},
    {DATASET_ID: float(SUBSET_SIZE)},
    shuffle=SUBSET_SIZE,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

# Marin-8B uses Llama 3 architecture. Extend to 32K context.
llama_8b_32k = dataclasses.replace(llama_8b, max_seq_len=32768)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "")

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=marin_tokenizer,
    initialize_from_hf="marin-community/marin-8b-base",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    max_seq_len=32768,
    seed=42,
    steps_per_checkpoint=max(1, NUM_TRAIN_STEPS // 4),
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,  # No weight decay for SFT (model already has pretraining WD effects)
    beta1=0.9,
    beta2=0.95,
    max_grad_norm=1.0,
)

subset_tag = f"{SUBSET_SIZE // 1000}k"
exp4898_sft = default_sft(
    name=f"exp4898_sft_marin_8b_swe_zero_{subset_tag}_32768tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=llama_8b_32k,
    sft_config=sft_config,
    tags=[
        "llama",
        "marin-8b-base",
        "swe-zero",
        "sft",
        subset_tag,
        RESOURCE_SUFFIX,
    ],
)

exp4898_checkpoint = exp4898_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print(f"=== exp4898: Marin-8B SWE-ZERO SFT ({subset_tag}) ===")
    print(f"Dataset: {DATASET_ID} ({SUBSET_SIZE:,} / {TOTAL_AVAILABLE:,} trajectories)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE} (elevated for Marin-8B; see #4225)")
    print(f"Resources: {RESOURCES.device.variant}")
    print(f"Checkpoint: {exp4898_checkpoint}")
    executor_main(steps=[exp4898_sft])
