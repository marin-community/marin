# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny rerank downstream-scaling eval on the dummy task."""

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
from experiments.downstream_scaling.evals.tasks.dummy import DummyTask, DummyTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.models import qwen2_5_7b

MODEL_KEY = "3e18"
SCORING_MODEL_KEY = "3e18"
TPU_TYPE = "v5p-8"

N_PROMPTS = 1
N_SAMPLES = 1
N_PROPOSALS = 2
NUM_WORKERS = 1
CHUNK_SIZE = 1

TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
PROPOSAL_LEN = 8
MAX_TOKENS = 16
SEED = 42
STOP_TOKENS = ("</s>", "<|im_end|>")


def make_task() -> DummyTask:
    return DummyTask(config=DummyTaskConfig(n_prompts=N_PROMPTS))


def make_algorithm(tpu_type: str) -> RerankCompletionAlgorithm:
    return RerankCompletionAlgorithm(
        config=RerankConfig(
            sampling=RerankSamplingConfig(
                n_samples=N_SAMPLES,
                n_proposals=N_PROPOSALS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                proposal_len=PROPOSAL_LEN,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            scoring_model_path=output_path_of(qwen2_5_7b),
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
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/dummy/rerank/{MODEL_KEY}/scorer_{SCORING_MODEL_KEY}",
            model_path=InputName.hardcoded(DELPHI_CHECKPOINTS[MODEL_KEY]),
            task=make_task(),
            alg=make_algorithm(tpu_type),
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-type", type=str, default=TPU_TYPE)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    executor_main(
        steps=build_steps(args.tpu_type),
        description="Tiny rerank downstream-scaling eval on the dummy task.",
    )
