# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Visual-only pre-training on the EU cluster.

Trains a Qwen3-architecture model from scratch on visual-only sequences:
  <|vision_start|> V₁..V₅₇₆ <|vision_end|> <|endoftext|>

Uses the pre-tokenized visual-only cache at:
  gs://marin-vlm-eu/hf_85m_levanter_cache_v2/visual_only

Environment variables:
    NUM_STEPS    — Total training steps (default: 10000)
    TPU_TYPE     — TPU slice type (default: "v4-64")
    EXP_NAME     — Experiment name for W&B (default: "visual-only-qwen3-1.7b")
    MUON_LR      — Muon learning rate (default: 0.004)
    ADAM_LR      — Adam learning rate for non-Muon params (default: 0.0012)
    LR_SCHEDULE  — Learning rate schedule: "cosine" or "constant" (default: "cosine")
    Z_LOSS_WEIGHT — Auxiliary z-loss weight (default: 0.0)

Usage:
    uv run python -m marin.run.ray_run \
        --cluster infra/marin-eu-west4.yaml \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e TPU_TYPE v4-64 \
        -e NUM_STEPS 10000 \
        -e EXP_NAME my-visual-run \
        -e MUON_LR 0.004 \
        -e ADAM_LR 0.0012 \
        -e LR_SCHEDULE cosine \
        -e Z_LOSS_WEIGHT 0.0 \
        -- python experiments/unified/visual_only_pretrain_eu.py
"""

import os

from experiments.unified.unified_pretrain_demo_eu import (
    _demo_train_config,
    visual_only_data_config,
)
from experiments.defaults import default_train
from experiments.qwen3 import qwen3_0_6b, qwen3_1_7b, qwen3_4b
from marin.execution import executor_main

import dataclasses

# --- Constants ---

VISUAL_ONLY_CACHE_PATH = "gs://marin-vlm-eu/hf_85m_levanter_cache_v2/visual_only"

# --- Environment Variables ---

NUM_STEPS = int(os.environ.get("NUM_STEPS", "10000"))
TPU_TYPE = os.environ.get("TPU_TYPE", "v4-64")
EXP_NAME = os.environ.get("EXP_NAME", "")
MUON_LR = float(os.environ.get("MUON_LR", "0.004"))
ADAM_LR = float(os.environ.get("ADAM_LR", "0.0012"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
Z_LOSS_WEIGHT = float(os.environ.get("Z_LOSS_WEIGHT", "0.0"))


# --- Experiment Step Factories ---


def make_visual_only_0_6b(
    muon_lr: float = 0.008,
    adam_lr: float = 0.0024,
    num_train_steps: int = NUM_STEPS,
    lr_schedule: str = "cosine",
    eval_benchmarks: list[str] | None = None,
    z_loss_weight: float = 0.0,
):
    step = default_train(
        name=EXP_NAME or "visual-only-qwen3-0.6b",
        tokenized=visual_only_data_config(
            visual_cache_path=VISUAL_ONLY_CACHE_PATH,
            eval_benchmarks=eval_benchmarks,
        ),
        model_config=qwen3_0_6b,
        train_config=_demo_train_config(
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            num_train_steps=num_train_steps,
            lr_schedule=lr_schedule,
            z_loss_weight=z_loss_weight,
        ),
        tags=["visual-only", "scaling", "qwen3", "0.6b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return step


def make_visual_only_1_7b(
    muon_lr: float = 0.004,
    adam_lr: float = 0.0012,
    num_train_steps: int = NUM_STEPS,
    lr_schedule: str = "cosine",
    eval_benchmarks: list[str] | None = None,
    z_loss_weight: float = 0.0,
):
    step = default_train(
        name=EXP_NAME or "visual-only-qwen3-1.7b",
        tokenized=visual_only_data_config(
            visual_cache_path=VISUAL_ONLY_CACHE_PATH,
            eval_benchmarks=eval_benchmarks,
        ),
        model_config=qwen3_1_7b,
        train_config=_demo_train_config(
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            num_train_steps=num_train_steps,
            lr_schedule=lr_schedule,
            z_loss_weight=z_loss_weight,
        ),
        tags=["visual-only", "scaling", "qwen3", "1.7b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return step


def make_visual_only_4b(
    muon_lr: float = 0.002,
    adam_lr: float = 0.0006,
    num_train_steps: int = NUM_STEPS,
    lr_schedule: str = "cosine",
    eval_benchmarks: list[str] | None = None,
    z_loss_weight: float = 0.0,
):
    step = default_train(
        name=EXP_NAME or "visual-only-qwen3-4b",
        tokenized=visual_only_data_config(
            visual_cache_path=VISUAL_ONLY_CACHE_PATH,
            eval_benchmarks=eval_benchmarks,
        ),
        model_config=qwen3_4b,
        train_config=_demo_train_config(
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            num_train_steps=num_train_steps,
            lr_schedule=lr_schedule,
            z_loss_weight=z_loss_weight,
        ),
        tags=["visual-only", "scaling", "qwen3", "4b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return step


if __name__ == "__main__":
    steps = [
        make_visual_only_0_6b(
            muon_lr=MUON_LR,
            adam_lr=ADAM_LR,
            num_train_steps=NUM_STEPS,
            lr_schedule=LR_SCHEDULE,
            z_loss_weight=Z_LOSS_WEIGHT,
        )
    ]
    executor_main(
        steps,
        description="Visual-only pre-training with Qwen3 architecture (EU)",
    )
