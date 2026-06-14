# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run IID downstream-scaling evals on masked GSM8K for the Delphi ladder.

v2 of ``run_delphi_masked_gsm8k_iid``: instead of one shared mask per problem,
each rollout gets an independently sampled mask. The task emits ``N_SAMPLES``
masked variants per problem and the IID algorithm draws a single completion per
variant.
"""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, output_path_of
from rigging.filesystem import marin_region

from experiments.downstream_scaling.evals.algorithms.iid import (
    IIDCompletionAlgorithm,
    IIDConfig,
    IIDExecutionConfig,
    IIDSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k_masked_iid import MaskedGSM8KIIDTask, MaskedGSM8KIIDTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS
from experiments.llama import llama3_tokenizer

N_SAMPLES = 64
N_PROBLEMS = 256
NUM_WORKERS = 1
TPU_TYPES: tuple[str, ...] = ("v5p-8",)

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


def make_task(mask_fraction: float) -> MaskedGSM8KIIDTask:
    return MaskedGSM8KIIDTask(
        config=MaskedGSM8KIIDTaskConfig(
            tokenizer_path=llama3_tokenizer,
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
            n_masks=N_SAMPLES,
            mask_fraction=mask_fraction,
            mask_text=MASK_TEXT,
        )
    )


def make_algorithm(tpu_types: list[str], region: str) -> IIDCompletionAlgorithm:
    return IIDCompletionAlgorithm(
        config=IIDConfig(
            sampling=IIDSamplingConfig(
                n_samples=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            execution=IIDExecutionConfig(
                num_workers=NUM_WORKERS,
                worker_resources=ResourceConfig.with_tpu(tpu_types, regions=[region]),
            ),
        )
    )


def build_steps(tpu_types: list[str], region: str):
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/delphi/masked_gsm8k/iid_v2/mask_{i:02d}/{slug}",
            model_path=output_path_of(checkpoint),
            task=make_task(mask_fraction),
            alg=make_algorithm(tpu_types, region),
        )
        for i, mask_fraction in enumerate(MASK_FRACTIONS)
        for slug, checkpoint in DELPHI_HF_DOWNLOADS.items()
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
        description="Delphi scaling-ladder IID evals on masked GSM8K with per-rollout masks.",
    )
