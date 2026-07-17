# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run CruxEval input and output prediction against brokered Qwen3 vLLM.

``CRUXEVAL_RESULTS`` is a lazy artifact containing lm-eval metrics and samples.
The evaluator runs through an isolated ``uv run`` environment in a non-preemptible
Iris CPU parent. Inference runs in TPU child jobs connected through Marin's
inference broker. CruxEval is scored by executing the predicted assertions rather
than by an LLM judge.

\b
Examples:
  uv run python experiments/evals/served_qwen3_cruxeval.py
  uv run python experiments/evals/served_qwen3_cruxeval.py --priority production
  uv run python experiments/evals/served_qwen3_cruxeval.py --local
"""

from dataclasses import replace

from fray.types import ResourceConfig
from marin.evaluation.lm_eval import LmEvalRun
from marin.execution.lazy import Artifact, ArtifactStep

from experiments.evals.served_lm_eval import (
    VLLM_WORKER_ENV_VARS,
    ServedLmEvalBenchmark,
    brokered_lm_eval_step,
    brokered_vllm_config,
    served_lm_eval_command,
)

CRUXEVAL_TASKS = ("cruxeval_input", "cruxeval_output")
CRUXEVAL_REGION = "us-west4"


def cruxeval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    inference = brokered_vllm_config(
        model="Qwen/Qwen3-0.6B-Base",
        tokenizer="Qwen/Qwen3-0.6B",
        workers=1,
        num_concurrent=16,
        timeout_seconds=None,
    )
    inference = replace(
        inference,
        worker_resources=ResourceConfig.with_tpu(
            "v5litepod-4",
            ram="96g",
            regions=[CRUXEVAL_REGION],
        ),
        worker_env_vars=VLLM_WORKER_ENV_VARS,
    )
    eval_run = LmEvalRun(
        tasks=CRUXEVAL_TASKS,
        output_path="/tmp/served-qwen3-cruxeval",
        limit=limit,
        confirm_run_unsafe_code=True,
        extra_model_args={
            "num_concurrent": inference.workers.max_in_flight_per_worker,
            "timeout": int(inference.proxy.request_timeout_seconds),
        },
    )
    return brokered_lm_eval_step(
        inference,
        eval_run,
        name="evals/qwen3-0.6b/cruxeval",
        version=version,
        parent_resources=ResourceConfig.with_cpu(
            cpu=0.5,
            ram="6g",
            disk="16g",
            regions=[CRUXEVAL_REGION],
            preemptible=False,
        ),
        parent_env_vars={},
    )


CRUXEVAL_RESULTS = cruxeval_step(version="2026.07.16")

main = served_lm_eval_command(
    __doc__,
    ServedLmEvalBenchmark(
        tasks=CRUXEVAL_TASKS,
        output_path="/tmp/served-qwen3-cruxeval",
        job_name="served-qwen3-cruxeval",
        confirm_run_unsafe_code=True,
    ),
)

if __name__ == "__main__":
    main()
