# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run rerank downstream-scaling evals on GSM8K for the Delphi ladder."""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import InputName, executor_main, output_path_of

from experiments.downstream_scaling.evals.algorithms.rerank import (
    RerankCompletionAlgorithm,
    RerankConfig,
    RerankExecutionConfig,
    RerankProposalConfig,
    RerankSamplingConfig,
    RerankScorerConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.models import qwen2_5_7b

N_SAMPLES = 32
N_PROBLEMS = 256
NUM_WORKERS = 1
CHUNK_SIZE = 64
TPU_TYPE = "v5p-8"

TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
PROPOSAL_LEN = 2
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
CANDIDATE_COUNTS = (2, 4, 8, 16, 32)


def make_task() -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def make_algorithm(tpu_type: str, candidate_count: int, scoring_model_path: InputName) -> RerankCompletionAlgorithm:
    return RerankCompletionAlgorithm(
        config=RerankConfig(
            sampling=RerankSamplingConfig(
                n_samples=N_SAMPLES,
                n_proposals=candidate_count,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                proposal_len=PROPOSAL_LEN,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            scoring_model_path=scoring_model_path,
            scorer=RerankScorerConfig(),
            proposal=RerankProposalConfig(),
            execution=RerankExecutionConfig(
                num_workers=NUM_WORKERS,
                chunk_size=CHUNK_SIZE,
                worker_resources=ResourceConfig.with_tpu(tpu_type),
                scorer_actor_resources=ResourceConfig.with_tpu(tpu_type),
            ),
        )
    )


def build_steps(tpu_type: str):
    scoring_model_path = output_path_of(qwen2_5_7b)
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/delphi/gsm8k/rerank/candidates_{candidate_count:02d}/{slug}",
            model_path=InputName.hardcoded(checkpoint),
            task=make_task(),
            alg=make_algorithm(tpu_type, candidate_count, scoring_model_path),
        )
        for candidate_count in CANDIDATE_COUNTS
        for slug, checkpoint in DELPHI_CHECKPOINTS.items()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-type", type=str, default=TPU_TYPE)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    executor_main(
        steps=build_steps(args.tpu_type),
        description="Delphi scaling-ladder rerank evals on GSM8K.",
    )
