# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run IID downstream-scaling evals on masked GSM8K for the Delphi ladder."""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import InputName, executor_main

from experiments.downstream_scaling.evals.algorithms.iid import (
    IIDCompletionAlgorithm,
    IIDConfig,
    IIDExecutionConfig,
    IIDSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k_masked import MaskedGSM8KTask, MaskedGSM8KTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.llama import llama3_tokenizer

N_SAMPLES = 32
N_PROBLEMS = 256
NUM_WORKERS = 1
TPU_TYPE = "v5p-8"

TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
MASK_FRACTIONS = tuple(i / 10 for i in range(11))
MASK_TEXT = "<mask>"


def make_task(mask_fraction: float) -> MaskedGSM8KTask:
    return MaskedGSM8KTask(
        config=MaskedGSM8KTaskConfig(
            tokenizer_path=llama3_tokenizer,
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
            mask_fraction=mask_fraction,
            mask_text=MASK_TEXT,
        )
    )


def make_algorithm(tpu_type: str) -> IIDCompletionAlgorithm:
    return IIDCompletionAlgorithm(
        config=IIDConfig(
            sampling=IIDSamplingConfig(
                n_samples=N_SAMPLES,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            execution=IIDExecutionConfig(
                num_workers=NUM_WORKERS,
                worker_resources=ResourceConfig.with_tpu(tpu_type),
            ),
        )
    )


def build_steps(tpu_type: str):
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/delphi/masked_gsm8k/iid/mask_{i:02d}/{slug}",
            model_path=InputName.hardcoded(checkpoint),
            task=make_task(mask_fraction),
            alg=make_algorithm(tpu_type),
        )
        for i, mask_fraction in enumerate(MASK_FRACTIONS)
        for slug, checkpoint in DELPHI_CHECKPOINTS.items()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-type", type=str, default=TPU_TYPE)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    executor_main(
        steps=build_steps(args.tpu_type),
        description="Delphi scaling-ladder IID evals on masked GSM8K.",
    )
