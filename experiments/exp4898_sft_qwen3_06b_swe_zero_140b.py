# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B-Base SFT on the full 140B SWE-ZERO trajectories dataset.

One epoch over the entire `AlienKevin/SWE-ZERO-140B-trajectories` HF dataset
(uploaded once the swarm pipeline at issue #4898 finishes). Saves a checkpoint
every NUM_TRAIN_STEPS // 4 -> checkpoints at 25%, 50%, 75%, 100% of training,
so swe-bench-100-random-folders evals can be triggered at each.

Differs from `exp4898_sft_marin_8b_swe_zero.py` in:
  - Qwen3-0.6B-Base architecture (head_dim=128 variant matching the HF config),
    using the Qwen3 tokenizer instead of the Marin tokenizer.
  - Trains over ALL trajectories (no SUBSET subsampling).
  - Defaults to v5p-16 in us-central1; the smaller model fits comfortably and
    runs faster than the 8B config.

Tracked in: https://github.com/marin-community/marin/issues/4898

Usage:
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-16 \
        --region us-central1 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e HF_TOKEN ${HF_TOKEN} \
        --no-wait \
        -- python experiments/exp4898_sft_qwen3_06b_swe_zero_140b.py
"""

import dataclasses
import math
import os

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    InstructionDatasetConfig,
    multi_turn_adapter,
)
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# Dataset built by experiments/swe_zero/upload_trajectories_12m.py from the
# 1260-shard per_pr/ swarm output (us-east5 cutoff snapshot, 51 % PR coverage).
DATASET_ID = "AlienKevin/SWE-ZERO-12M-trajectories"

# Projected post-round-3 rollout count (<=100/PR cap applied at aggregator):
# ~66.7K touched PRs x ~99 rollouts/PR avg ~ 6.6 M. The cap matters because
# pre-cap regional parquets carry ~10 M rollouts (baseline+multi-region overlap
# from the 2026-05-08 migration). Override via TRAIN_DATASET_SIZE env var when
# the round-3 SUMMARY lands the exact number - the constant just sets the
# cosine-schedule horizon when no override is provided.
# Historical context: round 1 = 6,517,753 (2026-05-05 Phase 2 upload);
# round 2 = 5,401,410 (HF dataset 2026-05-08 04:48 UTC).
DATASET_SIZE = int(os.environ.get("TRAIN_DATASET_SIZE", "6_600_000"))

# Qwen3 tokenizer - base model is Qwen3-0.6B-Base.
QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
QWEN3_INIT = "Qwen/Qwen3-0.6B-Base"

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-16")
# us-central1 v5p-{8,16} is heavily contested with other tenants (larry, tonyhlee,
# moojink, michaelryan); us-east5 v5p-8 was probed available 2026-05-05 and is
# the fallback. Override via SFT_REGION env var.
SFT_REGION = os.environ.get("SFT_REGION", "us-central1")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, regions=[SFT_REGION])
_NUM_CHIPS = int(RESOURCES.device.variant.split("-")[-1]) // 2

# Hyperparams
TARGET_EPOCHS = 1
TRAIN_BATCH_SIZE = min(64, _NUM_CHIPS * 4)
MICROBATCH_SIZE = TRAIN_BATCH_SIZE
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * DATASET_SIZE / TRAIN_BATCH_SIZE))

# LR for fresh Qwen3-0.6B SFT - moderate, no Marin-8B-specific elevation needed.
LEARNING_RATE = 5e-5

# Register the SWE-ZERO 12M dataset.
# max_parallelism=8 is conservative; the dataset has 1260 parquet shards and
# the v4/v5 attempts at 128 zephyr workers blew past HF's 1000 req/5min free
# tier with `429 Client Error` on every shard. 8 workers x ~30 req/min/shard
# stays comfortably under the limit.
INSTRUCTION_DATASET_NAME_TO_CONFIG[DATASET_ID] = InstructionDatasetConfig(
    hf_dataset_id=DATASET_ID,
    revision="main",
    adapter=multi_turn_adapter(conversation_column="messages"),
    metadata_columns=["instance_id", "repo", "exit_status"],
    splits=["train"],
    max_parallelism=8,
)


# Pre-built DolmaConversationOutput JSONL.gz files at this GCS path bypass the
# Marin `transform_conversation` Zephyr step (which crashed at the coord pod
# 30 sec after spawn on this 1260-shard input across SFT v4-v15). Built by
# `scripts/parquet_to_dolma_jsonl.py` on 2026-05-06: 1260 shards, 6,517,753
# kept rows, 42.5 GB. With a direct path, `default_tokenize` uses
# `TokenizeConfig` (file paths) instead of `HfTokenizeConfig` (which would
# trigger transform_conversation again).
PREBUILT_JSONL_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl/data/*.jsonl.gz"


# Custom Qwen3-style chat template with `{% generation %}` Jinja markers around
# assistant content. Levanter's `ChatLmDatasetFormat(mask_user_turns=True)`
# requires these markers to know which tokens to compute loss on. The Qwen3
# default chat template doesn't include them. We tried `mask_user_turns=False`
# in v18-v19 but Levanter's tokenize-cache consolidate step still expects
# `assistant_masks/offsets/` to exist (FileNotFoundError otherwise).
QWEN3_CHAT_TEMPLATE_WITH_GENERATION = (
    "{% for message in messages -%}\n"
    "{%- if message['role'] == 'assistant' -%}"
    "<|im_start|>assistant\n"
    "{% generation %}{{ message['content'] }}<|im_end|>\n"
    "{% endgeneration %}"
    "{%- else -%}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{%- endif -%}"
    "{% endfor -%}"
)


def create_tokenization_step():
    return default_tokenize(
        name="swe_zero_140b_qwen3_tokenizer",
        dataset=PREBUILT_JSONL_GLOB,
        tokenizer=QWEN3_TOKENIZER,
        format=ChatLmDatasetFormat(chat_template=QWEN3_CHAT_TEMPLATE_WITH_GENERATION),
    )


tokenized = create_tokenization_step()

mixture_config = lm_mixture_data_config(
    {DATASET_ID: tokenized},
    {DATASET_ID: float(DATASET_SIZE)},
    shuffle=DATASET_SIZE,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

# Qwen3-0.6B-Base, extended to 32K context to match SWE-ZERO generation.
qwen3_0_6b_32k = dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=32768)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "")

# Checkpoint cadence:
#   - HF_KEEP_INTERVAL: persist an HF-format checkpoint every 25/50/75/100 % of training
#     (these become the artifacts evaluated by parallel swe-bench-100-random-folders jobs).
#   - CHECKPOINT_INTERVAL: write Levanter resume-checkpoints every 25 steps so the run can
#     survive preemption on preemptible TPUs. v20 was preempted by bizon at 40 min in
#     (step ~thousands) with our previous interval of NUM_TRAIN_STEPS//4=101840 - that
#     loses 100 % of progress per preemption. exp4760 (Marin-32B SFT) noted the same
#     problem and dropped from 214 -> 25; we mirror that.
CHECKPOINT_INTERVAL = 25
HF_KEEP_INTERVAL = max(1, NUM_TRAIN_STEPS // 4)

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=QWEN3_TOKENIZER,
    initialize_from_hf=QWEN3_INIT,
    # Qwen3 base model embedding is padded to 151,936 (divisible by 4 for TPU
    # efficiency) but the tokenizer reports len()=151,669. Without this flag
    # Levanter sizes the data Vocab axis to 151,669 and HF init expects 151,936;
    # mismatch raises ValueError at train_lm step (v13 failure 2026-05-08).
    pad_tokenizer_to_match_model=True,
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    max_seq_len=32768,
    seed=42,
    steps_per_checkpoint=CHECKPOINT_INTERVAL,
    steps_per_hf_export=HF_KEEP_INTERVAL,
    lr_schedule="cosine",
    warmup=0.05,
    decay=0.9,
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.95,
    max_grad_norm=1.0,
)

exp4898_qwen3 = default_sft(
    name=f"exp4898_sft_qwen3_06b_swe_zero_140b_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_0_6b_32k,
    sft_config=sft_config,
    tags=[
        "qwen3",
        "qwen3-0.6b-base",
        "swe-zero",
        "sft",
        "140b",
        RESOURCE_SUFFIX,
    ],
)

exp4898_qwen3_final = exp4898_qwen3.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp4898: Qwen3-0.6B-Base SWE-ZERO SFT (full 140B) ===")
    print(f"Dataset: {DATASET_ID} ({DATASET_SIZE:,} trajectories - 1 epoch)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Checkpoint every: {CHECKPOINT_INTERVAL:,} steps (4 evals: 25/50/75/100%)")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Resources: {RESOURCES.device.variant} ({TARGET_EPOCHS} epoch, region=us-central1)")
    print(f"Final checkpoint: {exp4898_qwen3_final}")
    executor_main(steps=[exp4898_qwen3])
