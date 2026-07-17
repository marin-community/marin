# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 on CruxEval and save metrics and samples.

``CRUXEVAL_RESULTS`` is a lazy artifact containing lm-eval metrics and samples.
CruxEval is scored by executing the predicted assertions rather than by an LLM
judge.

\b
Examples:
  uv run iris --cluster=marin job run --job-name qwen3-cruxeval --region us-west4 \
    --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3_cruxeval
"""

from marin.execution.lazy import Artifact, ArtifactStep, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.served_lm_eval import (
    ServedLmEvalBenchmark,
    served_lm_eval_step,
)

CRUXEVAL_BENCHMARK = ServedLmEvalBenchmark(
    tasks=("cruxeval_input", "cruxeval_output"),
    confirm_run_unsafe_code=True,
)


def cruxeval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    return served_lm_eval_step(
        CRUXEVAL_BENCHMARK,
        name="evals/qwen3-0.6b/cruxeval",
        version=version,
        limit=limit,
    )


CRUXEVAL_RESULTS = cruxeval_step(version="2026.07.16")

if __name__ == "__main__":
    StepRunner().run([lower(CRUXEVAL_RESULTS)])
