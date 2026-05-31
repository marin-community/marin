# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run joint-decode downstream-scaling evals on GSM8K for the Delphi ladder.

Decoder (A) = each Delphi checkpoint in turn (RPA block-size patch on).
Advisor (B) = llama 3.1 8B (no patch). Sweeps top_k_a x top_k_b.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import InputName, executor_main, output_path_of

from experiments.downstream_scaling.evals.algorithms.joint_decode import (
    JointDecodeCompletionAlgorithm,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodeSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.models import llama_3_1_8b

N_SAMPLES = 1  # joint_decode is deterministic; must be 1
N_PROBLEMS = 256
NUM_WORKERS = 1
CHUNK_SIZE = 64
TPU_TYPES: tuple[str, ...] = ("v5p-8",)

# Generous enough to absorb the first-step XLA compilation skew between the
# two engines for the largest delphi sizes (60s was too short for 1e22).
BARRIER_TIMEOUT_S = 1200.0

MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TOP_K_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 16),
    (3, 16),
    (4, 16),
    (6, 16),
    (8, 16),
    (12, 16),
    (16, 16),
    (1, 1),
    (8, 1),
    (8, 2),
    (8, 4),
    (8, 8),
    (2, 1),
    (3, 1),
    (4, 1),
    (6, 1),
    (8, 1),
    (12, 1),
    (16, 1),
    (2, 2),
    (3, 2),
    (4, 2),
    (6, 2),
    (8, 2),
    (12, 2),
    (16, 2),
)

# 1e23 doesn't fit on a single TPU chip at TP=1; tensor-parallelism support
# is on the todo list. Skip for now.
SKIP_DELPHI_KEYS = frozenset({"1e23"})

# Microbatch the largest delphi sizes whose A/B vLLM schedulers otherwise
# desync; absent keys → None (whole chunk in one round-trip).
MICROBATCH_SIZE_BY_DELPHI_KEY: dict[str, int] = {"1e22": 8}


def make_task() -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def make_algorithm(
    tpu_types: list[str],
    region: str,
    top_k_a: int,
    top_k_b: int,
    advisor_model_path,
    microbatch_size: int | None,
) -> JointDecodeCompletionAlgorithm:
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=N_SAMPLES,
                max_tokens=MAX_TOKENS,
                top_k_a=top_k_a,
                top_k_b=top_k_b,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            advisor_model=JointDecodeModelConfig(),
            execution=JointDecodeExecutionConfig(
                num_workers=NUM_WORKERS,
                worker_resources=dataclasses.replace(ResourceConfig.with_tpu(tpu_types), regions=[region]),
                chunk_size=CHUNK_SIZE,
                microbatch_size=microbatch_size,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
            ),
        )
    )


def build_steps(tpu_types: list[str], region: str):
    advisor_model_path = output_path_of(llama_3_1_8b)
    return [
        make_eval_step(
            name=(f"downstream_scaling/evals/delphi/gsm8k/joint_decode/" f"topk_a{top_k_a:02d}_b{top_k_b:02d}/{slug}"),
            model_path=InputName.hardcoded(checkpoint),
            task=make_task(),
            alg=make_algorithm(
                tpu_types,
                region,
                top_k_a,
                top_k_b,
                advisor_model_path,
                MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug),
            ),
        )
        for top_k_a, top_k_b in TOP_K_PAIRS
        for slug, checkpoint in DELPHI_CHECKPOINTS.items()
        if slug not in SKIP_DELPHI_KEYS
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-types", nargs="+", default=list(TPU_TYPES))
    parser.add_argument("--region", type=str, required=True)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    executor_main(
        steps=build_steps(args.tpu_types, args.region),
        description="Delphi scaling-ladder joint-decode evals on GSM8K (top_k_a x top_k_b sweep).",
    )
