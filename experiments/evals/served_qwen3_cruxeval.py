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

from dataclasses import replace

from fray.types import ResourceConfig
from marin.evaluation.lm_eval import LmEvalRun
from marin.execution.lazy import Artifact, ArtifactStep, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.served_lm_eval import (
    brokered_lm_eval_step,
    brokered_vllm_config,
)

QWEN3_INFERENCE = brokered_vllm_config(
    model="Qwen/Qwen3-0.6B-Base",
    tokenizer="Qwen/Qwen3-0.6B",
    worker_resources=ResourceConfig.with_tpu("v5litepod-4", ram="96g", regions=["us-west4"]),
)
EVAL_PARENT_RESOURCES = ResourceConfig.with_cpu(
    cpu=0.5,
    ram="6g",
    disk="16g",
    regions=["us-west4"],
    preemptible=False,
)
CRUXEVAL_RUN = LmEvalRun(
    tasks=("cruxeval_input", "cruxeval_output"),
    confirm_run_unsafe_code=True,
)


def cruxeval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    return brokered_lm_eval_step(
        QWEN3_INFERENCE,
        replace(CRUXEVAL_RUN, limit=limit),
        name="evals/qwen3-0.6b/cruxeval",
        version=version,
        parent_resources=EVAL_PARENT_RESOURCES,
        parent_env_vars={},
    )


CRUXEVAL_RESULTS = cruxeval_step(version="2026.07.16")

if __name__ == "__main__":
    StepRunner().run([lower(CRUXEVAL_RESULTS)])
