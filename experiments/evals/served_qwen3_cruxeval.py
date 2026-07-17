# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run CruxEval input and output prediction against brokered Qwen3 vLLM.

``CRUXEVAL_RESULTS`` is a lazy artifact containing lm-eval metrics and samples.
By default, the evaluator runs through an isolated ``uv run`` environment in a
non-preemptible Iris CPU parent, with inference in TPU child jobs connected through
Marin's inference broker. ``--local`` instead runs the broker, proxy, and worker in
the current process. CruxEval is scored by executing the predicted assertions
rather than by an LLM judge.

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

CRUXEVAL_BENCHMARK = ServedLmEvalBenchmark(
    tasks=("cruxeval_input", "cruxeval_output"),
    output_path="/tmp/served-qwen3-cruxeval",
    job_name="served-qwen3-cruxeval",
    confirm_run_unsafe_code=True,
)


def cruxeval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    inference = brokered_vllm_config(
        model=CRUXEVAL_BENCHMARK.model,
        tokenizer=CRUXEVAL_BENCHMARK.tokenizer,
        workers=CRUXEVAL_BENCHMARK.workers,
        num_concurrent=CRUXEVAL_BENCHMARK.num_concurrent,
        timeout_seconds=None,
    )
    inference = replace(
        inference,
        worker_resources=ResourceConfig.with_tpu(
            CRUXEVAL_BENCHMARK.tpu_type,
            ram=CRUXEVAL_BENCHMARK.worker_ram,
            regions=[CRUXEVAL_BENCHMARK.region] if CRUXEVAL_BENCHMARK.region else None,
        ),
        worker_env_vars=dict(VLLM_WORKER_ENV_VARS),
    )
    eval_run = LmEvalRun(
        tasks=CRUXEVAL_BENCHMARK.tasks,
        output_path=CRUXEVAL_BENCHMARK.output_path,
        limit=limit,
        confirm_run_unsafe_code=CRUXEVAL_BENCHMARK.confirm_run_unsafe_code,
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
            ram=CRUXEVAL_BENCHMARK.parent_ram,
            disk="16g",
            regions=[CRUXEVAL_BENCHMARK.region] if CRUXEVAL_BENCHMARK.region else None,
            preemptible=False,
        ),
        parent_env_vars={},
    )


CRUXEVAL_RESULTS = cruxeval_step(version="2026.07.16")

main = served_lm_eval_command(
    __doc__,
    CRUXEVAL_BENCHMARK,
)

if __name__ == "__main__":
    main()
