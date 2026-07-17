# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 on HumanEval as a lazy artifact.

\b
Examples:
  uv run iris --cluster=marin job run --job-name qwen3-humaneval --region us-west4 \
    --cpu 1 --memory 2G --extra cpu --priority interactive --no-wait \
    -- python -m experiments.evals.served_qwen3_humaneval
"""

from dataclasses import replace

from fray.types import ResourceConfig
from marin.evaluation.lm_eval import LmEvalRun
from marin.execution.lazy import Artifact, ArtifactStep, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.served_lm_eval import brokered_lm_eval_step, brokered_vllm_config

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
        parent_resources=EVAL_PARENT_RESOURCES,
        parent_env_vars={"HF_ALLOW_CODE_EVAL": "1"},
    )


HUMANEVAL_RESULTS = humaneval_step(version="2026.07.17")

if __name__ == "__main__":
    StepRunner().run([lower(HUMANEVAL_RESULTS)])
