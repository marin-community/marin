# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Idealized Ferry baseline mirroring docs/reports/marin-32b-retro.md."""

import math

from experiments.defaults import SimpleTrainConfig, default_train
from experiments.qwen3 import qwen3_1_7b, qwen3_8b
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets import (
    NEMOTRON_WEIGHTS,
    tokenize_nemotron,
)
from experiments.exp934_hq_vs_pt import pt_vs_hq_components
from experiments.midtraining_datasets import (
    megamath_token_counts,
    megamath_tokenized,
    stackv2_edu_filtered_python_tokenized,
)
from experiments.tootsie.exp600_tootsie import phase_3_tokenized, starling_components
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

SEQ_LEN = 4096
BATCH_SIZE_1B = 128
BATCH_SIZE_8B = 1024

NUM_1B_TRAIN_TOKENS = int(34e9)  # 34 billion tokens
NUM_8B_TRAIN_TOKENS = int(160e9)  # 160 billion tokens

NUM_1B_TRAIN_STEPS = NUM_1B_TRAIN_TOKENS // (BATCH_SIZE_1B * SEQ_LEN)
NUM_8B_TRAIN_STEPS = NUM_8B_TRAIN_TOKENS // (BATCH_SIZE_8B * SEQ_LEN)

train_config_1b = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-32"),
    train_batch_size=BATCH_SIZE_1B,
    num_train_steps=NUM_1B_TRAIN_STEPS,
    learning_rate=3e-3,
    # Use WSD (warmup-stable-decay) with linear decay
    lr_schedule="linear",
    decay=0.2,  # 20% of steps for decay/cooldown within WSD
    weight_decay=0.033,
    min_lr_ratio=0.1,
    warmup=5000,
    z_loss_weight=5e-6,
)

train_config_8b = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-128"),
    train_batch_size=BATCH_SIZE_8B,
    num_train_steps=NUM_8B_TRAIN_STEPS,
    learning_rate=2e-3,
    # Use WSD (warmup-stable-decay) with linear decay
    lr_schedule="linear",
    decay=0.2,  # 20% of steps for decay/cooldown within WSD
    weight_decay=0.05,
    min_lr_ratio=0.1,
    warmup=5000,
    z_loss_weight=5e-6,
)

# Build a varying mixture schedule similar to the Tootsie runs
nemotron_steps = tokenize_nemotron()
proofpile_2_step = dclm_components_llama3["proofpile_2"]
starcoder_step = dclm_components_llama3["starcoderdata"]

# Phase 1 (PT): Nemotron + Starcoder + ProofPile-2
NEMOTRON_PT_MIX_WEIGHTS = {
    **NEMOTRON_WEIGHTS,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}


def _build_varying_mixture_for_steps(
    num_train_steps: int,
    *,
    train_batch_size: int,
    mixture_block_size: int = 2048,
):
    """
    Build the varying data mixture with phase cutovers proportional to the
    Tootsie schedules (160k/192k and 174k/192k) for a given training horizon.
    """
    # Allocate 20% of steps to the cooldown (midtraining) phase
    requested_cooldown_start_step = max(1, int(num_train_steps * 0.8))

    # MixtureDataset requires stage boundaries to fall on sequence-block boundaries.
    # Because train_set() rescales step indices to sequence indices using the batch schedule,
    # we align the step index so that (step * train_batch_size) is a multiple of mixture_block_size.
    step_multiple = mixture_block_size // math.gcd(mixture_block_size, train_batch_size)
    cooldown_start_step = (requested_cooldown_start_step // step_multiple) * step_multiple
    if cooldown_start_step == 0:
        cooldown_start_step = step_multiple

    # Two-phase schedule only: pretraining then midtraining.
    # StackV2 EDU Python is included from the start of midtraining,
    # instead of being introduced as a later third phase.
    return lm_varying_mixture_data_config(
        components={
            **nemotron_steps,
            "starcoderdata": starcoder_step,
            "proofpile_2": proofpile_2_step,
            **phase_3_tokenized,
            **{k: v for k, v in pt_vs_hq_components.items() if k != "all_math"},
            **megamath_tokenized,
            **starling_components,
            STACKV2_EDU_PYTHON_KEY: stackv2_edu_filtered_python_tokenized,
        },
        weights_list=[
            (0, NEMOTRON_PT_MIX_WEIGHTS),  # Pretraining
            (cooldown_start_step, mantis_cooldown_weights_with_stackv2_python),  # Midtraining
        ],
        mixture_block_size=mixture_block_size,
    )


# Phase 2 (Cooldown): 70% PT mix, 30% HQ with MegaMath split
HQ_COOLDOWN_WEIGHTS = {
    "dolmino/flan": 0.017 * 2,
    "dolmino/pes2o": 0.0581,
    "dolmino/stackexchange": 0.0171,
    "dolmino/wiki": 0.00365,
    "all_math": 0.371,
    "arxiv_markdownified": 0.0581,
    "stackexchange_custom": 0.0171,
    "wikipedia_markdown": 0.00365,
    "medu_science_qa": 0.0012,
    "finemath-3-plus": 0.034,
}

nemotron_total = sum(NEMOTRON_PT_MIX_WEIGHTS.values())
all_math_weight = HQ_COOLDOWN_WEIGHTS["all_math"]
megamath_total = sum(megamath_token_counts.values())

mantis_hq_cooldown_weights = {
    **{k: v for k, v in HQ_COOLDOWN_WEIGHTS.items() if k != "all_math"},
    **{
        split: (all_math_weight if split != "megamath/web" else all_math_weight / 4) * weight / megamath_total
        for split, weight in megamath_token_counts.items()
    },
}

mantis_total_hq_weight = sum(mantis_hq_cooldown_weights.values())

mantis_cooldown_weights = {
    **{k: v * 0.7 / nemotron_total for k, v in NEMOTRON_PT_MIX_WEIGHTS.items()},
    **{k: v * (0.3 / mantis_total_hq_weight) for k, v in mantis_hq_cooldown_weights.items()},
}

STACKV2_EDU_PYTHON_KEY = "common_pile_stackv2_edu_filtered_python"
STACKV2_EDU_PYTHON_WEIGHT = 0.01463

mantis_hq_cooldown_weights_with_stackv2_python = {
    **mantis_hq_cooldown_weights,
    STACKV2_EDU_PYTHON_KEY: STACKV2_EDU_PYTHON_WEIGHT,
}
mantis_total_hq_weight_with_stackv2_python = sum(mantis_hq_cooldown_weights_with_stackv2_python.values())

mantis_cooldown_weights_with_stackv2_python = {
    **{k: v * 0.7 / nemotron_total for k, v in NEMOTRON_PT_MIX_WEIGHTS.items()},
    **{
        k: v * (0.3 / mantis_total_hq_weight_with_stackv2_python)
        for k, v in mantis_hq_cooldown_weights_with_stackv2_python.items()
    },
}

varying_mixture_1b = _build_varying_mixture_for_steps(
    NUM_1B_TRAIN_STEPS, train_batch_size=BATCH_SIZE_1B, mixture_block_size=2048
)
varying_mixture_8b = _build_varying_mixture_for_steps(
    NUM_8B_TRAIN_STEPS, train_batch_size=BATCH_SIZE_8B, mixture_block_size=2048
)

ferry_model_1b = default_train(
    name="ferry_qwen3_1_7b_pt_to_cooldown",
    tokenized=varying_mixture_1b,
    model_config=qwen3_1_7b,
    train_config=train_config_1b,
    eval_harness_tasks=[],
    override_output_path="checkpoints/ferry_qwen3_1_7b_pt_to_cooldown-027563",
)

ferry_model_8b = default_train(
    name="ferry_qwen3_8b_pt_to_cooldown",
    tokenized=varying_mixture_8b,
    model_config=qwen3_8b,
    train_config=train_config_8b,
    eval_harness_tasks=[],
    override_output_path="checkpoints/ferry_qwen3_8b_pt_to_cooldown-1fc2cb",
)

# Main execution block
if __name__ == "__main__":
    executor_main(
        steps=[ferry_model_1b, ferry_model_8b],
        description=("Ferry: PT on Nemotron+Code then cooldown to HQ mix, scaled from Tootsie schedules"),
    )
