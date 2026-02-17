# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daily ferry template for the 125M-ish integration run.

This script is intentionally simple and serves as a living ferry template that
agents can update on a daily cadence with small, reviewable changes.

Expected workflow:
1. Propose a PR with bounded config updates.
2. Get human approval.
3. Launch with `ray_run.py` and monitor to completion.
"""

import datetime as dt
import os

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

from experiments.defaults import SimpleTrainConfig, default_train
from experiments.llama import compute_num_parameters, llama3_tokenizer_vocab_size, llama_150m
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3

# ---------------------------
# Daily ferry policy defaults
# ---------------------------
DEFAULT_MODEL_FLOPS_TARGET = int(1e19)
DEFAULT_CROSS_ENTROPY_BLOCK_SIZE = 2048
TRAIN_BATCH_SIZE = 512
TRAIN_SEQ_LEN = llama_150m.max_seq_len


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


# Approximate FLOPs model used for scaling guidance:
# total_flops ~= 6 * num_params * num_tokens
MODEL_FLOPS_TARGET = _int_env("FERRY_MODEL_FLOPS_TARGET", DEFAULT_MODEL_FLOPS_TARGET)
CROSS_ENTROPY_BLOCK_SIZE = _int_env("FERRY_CROSS_ENTROPY_BLOCK_SIZE", DEFAULT_CROSS_ENTROPY_BLOCK_SIZE)
NUM_MODEL_PARAMS = compute_num_parameters(llama_150m, llama3_tokenizer_vocab_size)
NUM_TRAIN_TOKENS = MODEL_FLOPS_TARGET // (6 * NUM_MODEL_PARAMS)
NUM_TRAIN_STEPS = _int_env(
    "FERRY_NUM_TRAIN_STEPS",
    max(1, NUM_TRAIN_TOKENS // (TRAIN_BATCH_SIZE * TRAIN_SEQ_LEN)),
)

# Agents can override date from the launch environment to force deterministic naming.
FERRY_DATE = os.environ.get("FERRY_DATE", dt.date.today().isoformat())
RUN_NAME = f"ferry_daily_125m_{FERRY_DATE}"

# Agent edit surface: keep daily changes small (usually 1-2 knobs).
train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=TRAIN_BATCH_SIZE,
    train_seq_len=TRAIN_SEQ_LEN,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=3e-3,
    lr_schedule="linear",
    decay=0.2,
    weight_decay=0.1,
    min_lr_ratio=0.1,
    warmup=1000,
    z_loss_weight=1e-4,
    cross_entropy_block_size=CROSS_ENTROPY_BLOCK_SIZE,
)

daily_ferry = default_train(
    name=RUN_NAME,
    tokenized=dclm_mixture_config_llama3,
    model_config=llama_150m,
    train_config=train_config,
    tags=["ferry", "daily", "integration", "125m"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(
        steps=[daily_ferry],
        description="Daily ferry (125M-ish): bounded-change TPU integration run.",
    )
