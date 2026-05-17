# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny joint-decode downstream-scaling eval on the dummy task.

Decoder (A) = delphi `3e18` (needs the RPA block-size patch).
Advisor (B) = llama 3.1 8B (no patch).
"""

from __future__ import annotations

import argparse
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
from experiments.downstream_scaling.evals.tasks.dummy import DummyTask, DummyTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.models import llama_3_1_8b

DECODER_MODEL_KEY = "3e18"
ADVISOR_MODEL_NAME = "llama_3_1_8b"
TPU_TYPE = "v5p-8"

N_PROMPTS = 1
N_SAMPLES = 1  # joint_decode is deterministic; must be 1
NUM_WORKERS = 1
CHUNK_SIZE = 1

TOP_K_A = 8
TOP_K_B = 8
MAX_TOKENS = 16
SEED = 42
STOP_TOKENS = ("</s>", "<|im_end|>")


def make_task() -> DummyTask:
    return DummyTask(config=DummyTaskConfig(n_prompts=N_PROMPTS))


def make_algorithm(tpu_type: str) -> JointDecodeCompletionAlgorithm:
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=N_SAMPLES,
                max_tokens=MAX_TOKENS,
                top_k_a=TOP_K_A,
                top_k_b=TOP_K_B,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=output_path_of(llama_3_1_8b),
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            advisor_model=JointDecodeModelConfig(),
            execution=JointDecodeExecutionConfig(
                num_workers=NUM_WORKERS,
                worker_resources=ResourceConfig.with_tpu(tpu_type),
                chunk_size=CHUNK_SIZE,
            ),
        )
    )


def build_steps(tpu_type: str):
    return [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/dummy/joint_decode/"
                f"{DECODER_MODEL_KEY}_advisor_{ADVISOR_MODEL_NAME}"
            ),
            model_path=InputName.hardcoded(DELPHI_CHECKPOINTS[DECODER_MODEL_KEY]),
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
        description="Tiny joint-decode downstream-scaling eval on the dummy task.",
    )
