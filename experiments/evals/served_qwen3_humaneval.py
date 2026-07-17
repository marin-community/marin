# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 on HumanEval as a lazy artifact.

\b
Examples:
  uv run iris --cluster=marin job run --job-name qwen3-humaneval --region us-west4 \
    --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3_humaneval
"""

from marin.execution.lazy import Artifact, ArtifactStep, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.served_lm_eval import ServedLmEvalBenchmark, served_lm_eval_step

HUMANEVAL_BENCHMARK = ServedLmEvalBenchmark(
    tasks=("humaneval",),
    confirm_run_unsafe_code=True,
    parent_env_vars=(("HF_ALLOW_CODE_EVAL", "1"),),
)


def humaneval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    return served_lm_eval_step(
        HUMANEVAL_BENCHMARK,
        name="evals/qwen3-0.6b/humaneval",
        version=version,
        limit=limit,
    )


HUMANEVAL_RESULTS = humaneval_step(version="2026.07.17")

if __name__ == "__main__":
    StepRunner().run([lower(HUMANEVAL_RESULTS)])
