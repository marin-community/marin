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

"""Idealized Ferry baseline for OLMoE 1B/7B (≈1.3B active / 7B total).

This mirrors `experiments/ferries/initial_ferry.py` (Nemotron+Code PT -> HQ cooldown mix),
but swaps the model config to an OLMoE-style MoE.
"""

import argparse
import math
import sys
import time

from experiments.defaults import SimpleTrainConfig, default_train
from experiments.exp934_hq_vs_pt import pt_vs_hq_components
from experiments.midtraining_datasets import (
    megamath_token_counts,
    megamath_tokenized,
    stackv2_edu_filtered_python_tokenized,
)
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.speedrun.custom_mixtral import MixtralConfig
from experiments.tootsie.exp600_tootsie import phase_3_tokenized, starling_components
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"

DEFAULT_SEQ_LEN = 4096
DEFAULT_GLOBAL_BATCH_SIZE = 128
DEFAULT_TOKEN_TARGET = int(34e9)  # 34B tokens (mirrors the "1B" horizon in initial_ferry.py)


def _parse_args():
    parser = argparse.ArgumentParser(description="OLMoE 1B/7B ferry launcher (PT -> cooldown).")
    parser.add_argument("--tpu-type", default="v5p-64")
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--token-target", type=int, default=DEFAULT_TOKEN_TARGET)
    parser.add_argument("--run-suffix", type=str, default=None)
    return parser.parse_known_args()


def _steps_for_token_target(token_target: int, global_batch_size: int, seq_len: int) -> int:
    return math.ceil(token_target / (global_batch_size * seq_len))


def _build_olmoe_1b7b_config(seq_len: int) -> MixtralConfig:
    return MixtralConfig(
        seq_len=seq_len,
        hidden_dim=2048,
        intermediate_dim=1024,
        num_layers=16,
        num_heads=16,
        num_kv_heads=8,
        n_routed_experts=64,
        num_experts_per_tok=8,
        layer_norm_epsilon=1e-5,
        gradient_checkpointing=True,
        scan_layers=True,
        use_gmm=True,
        cross_entropy_block_size=32000,
        reference_checkpoint=OLMOE_1B7B_REFERENCE_CHECKPOINT,
        tokenizer=OLMOE_1B7B_REFERENCE_CHECKPOINT,
    )


# ---------------------------------------------------------------------------
# Data mixture: mirrors experiments/ferries/initial_ferry.py
# ---------------------------------------------------------------------------
nemotron_steps = tokenize_nemotron()
proofpile_2_step = dclm_components_llama3["proofpile_2"]
starcoder_step = dclm_components_llama3["starcoderdata"]

NEMOTRON_PT_MIX_WEIGHTS = {
    **NEMOTRON_WEIGHTS,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}

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

STACKV2_EDU_PYTHON_KEY = "common_pile_stackv2_edu_filtered_python"
STACKV2_EDU_PYTHON_WEIGHT = 0.01463

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


def _build_varying_mixture_for_steps(
    num_train_steps: int,
    *,
    train_batch_size: int,
    mixture_block_size: int = 2048,
):
    # Allocate 20% of steps to the cooldown (midtraining) phase.
    requested_cooldown_start_step = max(1, int(num_train_steps * 0.8))

    step_multiple = mixture_block_size // math.gcd(mixture_block_size, train_batch_size)
    cooldown_start_step = (requested_cooldown_start_step // step_multiple) * step_multiple
    if cooldown_start_step == 0:
        cooldown_start_step = step_multiple

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
        permutation_type="linear",
        mixture_block_size=mixture_block_size,
    )


if __name__ == "__main__":
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    num_train_steps = _steps_for_token_target(args.token_target, args.global_batch_size, args.seq_len)
    varying_mixture = _build_varying_mixture_for_steps(num_train_steps, train_batch_size=args.global_batch_size)

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(args.tpu_type),
        train_batch_size=args.global_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=3e-3,
        lr_schedule="linear",
        decay=0.2,
        weight_decay=0.033,
        min_lr_ratio=0.1,
        warmup=5000,
        z_loss_weight=5e-6,
    )

    run_suffix = args.run_suffix or (
        f"olmoe_1b7b_ferry_bs{args.global_batch_size}_seq{args.seq_len}_{args.tpu_type}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )

    ferry_step = default_train(
        name=f"ferry_olmoe_1b7b_pt_to_cooldown_{run_suffix}",
        tokenized=varying_mixture,
        model_config=_build_olmoe_1b7b_config(args.seq_len),
        train_config=train_config,
        eval_harness_tasks=[],
    )

    executor_main(
        steps=[ferry_step],
        description="Ferry (OLMoE 1B/7B): PT on Nemotron+Code then cooldown to HQ mix, scaled from Tootsie schedules",
    )
