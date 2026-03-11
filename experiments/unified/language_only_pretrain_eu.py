# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Language-only pre-training on the EU cluster.

Trains a Qwen3-architecture model from scratch on text-only data (Nemotron-CC
hq_actual). Uses the unified tokenizer (144k vocab) for architecture
consistency with the unified and visual-only experiments.

Evaluates on:
  - Paloma + uncheatable eval validation sets (per-step val loss)
  - Text eval benchmarks: hellaswag, winogrande, arc_easy, arc_challenge, mmlu
  - CORE_TASKS via lm-evaluation-harness (task accuracy)

Environment variables:
    NUM_STEPS    — Total training steps (default: 10000)
    TPU_TYPE     — TPU slice type (default: "v4-64")
    EXP_NAME     — Experiment name for W&B (default: "language-only-qwen3-1.7b")
    MUON_LR      — Muon learning rate (default: 0.004)
    ADAM_LR      — Adam learning rate for non-Muon params (default: 0.0012)
    LR_SCHEDULE  — Learning rate schedule: "cosine" or "constant" (default: "cosine")
    Z_LOSS_WEIGHT — Auxiliary z-loss weight (default: 0.0)

Usage:
    uv run python -m marin.run.ray_run \\
        --cluster infra/marin-eu-west4.yaml \\
        -e WANDB_API_KEY ${WANDB_API_KEY} \\
        -e TPU_TYPE v4-64 \\
        -e NUM_STEPS 10000 \\
        -e EXP_NAME my-language-run \\
        -e MUON_LR 0.004 \\
        -e ADAM_LR 0.0012 \\
        -e LR_SCHEDULE cosine \\
        -e Z_LOSS_WEIGHT 0.0 \\
        -- python experiments/unified/language_only_pretrain_eu.py
"""

import dataclasses
import os

from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat

from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import tokenize_nemotron
from experiments.qwen3 import qwen3_1_7b
from experiments.unified.unified_pretrain_demo_eu import (
    UNIFIED_TOKENIZER_PATH,
    TEXT_EVAL_CACHE_PATH,
    _demo_train_config,
)
from marin.execution import executor_main
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

# --- Environment Variables ---

NUM_STEPS = int(os.environ.get("NUM_STEPS", "10000"))
TPU_TYPE = os.environ.get("TPU_TYPE", "v4-64")
EXP_NAME = os.environ.get("EXP_NAME", "")
MUON_LR = float(os.environ.get("MUON_LR", "0.004"))
ADAM_LR = float(os.environ.get("ADAM_LR", "0.0012"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
Z_LOSS_WEIGHT = float(os.environ.get("Z_LOSS_WEIGHT", "0.0"))

DEFAULT_TEXT_EVAL_BENCHMARKS = [
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]


# --- Data Config ---


def language_only_data_config(
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
    text_eval_cache_path: str = TEXT_EVAL_CACHE_PATH,
) -> LmDataConfig:
    """Data config for language-only training — Nemotron-CC hq_actual, no multimodal data.

    Text eval benchmarks are included as validation-only components (weight=0)
    for per-step val loss tracking. Paloma validation is added separately via
    use_default_validation=True in default_train.
    """
    nemotron_steps = tokenize_nemotron()
    hq_key = "nemotron_cc/hq_actual"
    components = {hq_key: step_to_lm_mixture_component(nemotron_steps[hq_key], include_raw_paths=True)}
    weights: dict[str, float] = {hq_key: 1.0}

    # Text eval benchmarks: weight 0.0 -> eval-only (not used in training).
    if text_eval_benchmarks is not None:
        text_prebuilt_format = PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
        )
        for bench in text_eval_benchmarks:
            components[f"text_eval_{bench}"] = DatasetComponent(
                cache_dir=f"{text_eval_cache_path}/{bench}",
                format=text_prebuilt_format,
                pack=True,
                tags=["eval", "text"],
            )
            weights[f"text_eval_{bench}"] = 0.0

    return LmDataConfig(
        tokenizer=UNIFIED_TOKENIZER_PATH,
        components=components,
        train_weights=weights,
        shuffle=True,
        permutation_type="feistel",
    )


# --- Experiment Step Factory ---


def make_language_only_1_7b(
    muon_lr: float = 0.004,
    adam_lr: float = 0.0012,
    num_train_steps: int = NUM_STEPS,
    lr_schedule: str = "cosine",
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
    z_loss_weight: float = 0.0,
):
    step = default_train(
        name=EXP_NAME or "language-only-qwen3-1.7b",
        tokenized=language_only_data_config(
            text_eval_benchmarks=text_eval_benchmarks,
        ),
        model_config=qwen3_1_7b,
        train_config=_demo_train_config(
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            num_train_steps=num_train_steps,
            lr_schedule=lr_schedule,
            z_loss_weight=z_loss_weight,
        ),
        tags=["language-only", "scaling", "qwen3", "1.7b"],
        eval_harness_tasks=CORE_TASKS,
        use_default_validation=True,
    )
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return step


if __name__ == "__main__":
    steps = [make_language_only_1_7b(
        muon_lr=MUON_LR,
        adam_lr=ADAM_LR,
        num_train_steps=NUM_STEPS,
        lr_schedule=LR_SCHEDULE,
        z_loss_weight=Z_LOSS_WEIGHT,
    )]
    executor_main(
        steps,
        description="Language-only pre-training with Qwen3 architecture (EU)",
    )
