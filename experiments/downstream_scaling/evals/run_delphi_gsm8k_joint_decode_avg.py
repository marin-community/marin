# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run joint-decode-avg downstream-scaling evals on GSM8K for the Delphi ladder.

Decoder (A) = each Delphi checkpoint in turn (RPA block-size patch on).
Advisor (B) = llama 3.1 8B (no patch). Sweeps advisor_weight.
"""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import InputName, executor_main, output_path_of
from rigging.filesystem import marin_region

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg import (
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

N_SAMPLES = 32
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

TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 16
ADVISOR_WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

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
    advisor_weight: float,
    advisor_model_path,
    microbatch_size: int | None,
) -> JointDecodeCompletionAlgorithm:
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=N_SAMPLES,
                max_tokens=MAX_TOKENS,
                top_k_a=TOP_K_A,
                top_k_b=TOP_K_B,
                seed=SEED,
                temperature=TEMPERATURE,
                advisor_weight=advisor_weight,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            advisor_model=JointDecodeModelConfig(),
            execution=JointDecodeExecutionConfig(
                num_workers=NUM_WORKERS,
                worker_resources=ResourceConfig.with_tpu(tpu_types, regions=[region]),
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
            name=(
                f"downstream_scaling/evals/delphi/gsm8k/joint_decode_avg/"
                f"advisor_weight{round(advisor_weight * 100):03d}/{slug}"
            ),
            model_path=InputName.hardcoded(checkpoint),
            task=make_task(),
            alg=make_algorithm(
                tpu_types,
                region,
                advisor_weight,
                advisor_model_path,
                MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug),
            ),
        )
        for advisor_weight in ADVISOR_WEIGHTS
        for slug, checkpoint in DELPHI_CHECKPOINTS.items()
        if slug not in SKIP_DELPHI_KEYS
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-types", nargs="+", default=list(TPU_TYPES))
    parser.add_argument("--region", type=str, default=None)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    region = args.region or marin_region()
    if region is None:
        parser.error("--region not given and not inferable from the environment")

    executor_main(
        steps=build_steps(args.tpu_types, region),
        description="Delphi scaling-ladder joint-decode-avg evals on GSM8K (advisor_weight sweep).",
    )
