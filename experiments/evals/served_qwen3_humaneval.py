# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 on HumanEval as a lazy artifact.

\b
Examples:
  uv run iris --cluster=marin job run --job-name qwen3-humaneval \
    --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3_humaneval
"""

from dataclasses import replace

from marin.evaluation.lm_eval import LmEvalRun
from marin.execution.lazy import Artifact, ArtifactStep, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.served_lm_eval import brokered_lm_eval_step
from experiments.evals.served_qwen3 import QWEN3_INFERENCE

HUMANEVAL_RUN = LmEvalRun(
    tasks=("humaneval",),
    confirm_run_unsafe_code=True,
)


def humaneval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    return brokered_lm_eval_step(
        QWEN3_INFERENCE,
        replace(HUMANEVAL_RUN, limit=limit),
        name="evals/qwen3-0.6b/humaneval",
        version=version,
        parent_env_vars={"HF_ALLOW_CODE_EVAL": "1"},
    )


HUMANEVAL_RESULTS = humaneval_step(version="2026.07.17")

if __name__ == "__main__":
    StepRunner().run([lower(HUMANEVAL_RESULTS)])
