# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate broker-served Qwen3 on CruxEval and save metrics and samples.

``CRUXEVAL_RESULTS`` is a lazy artifact containing lm-eval metrics and samples.
CruxEval is scored by executing the predicted assertions rather than by an LLM
judge.

\b
Examples:
  uv run python experiments/evals/served_qwen3_cruxeval.py
  uv run python experiments/evals/served_qwen3_cruxeval.py --priority production
  uv run python experiments/evals/served_qwen3_cruxeval.py --launcher local
"""

from fray.types import ResourceConfig
from marin.execution.lazy import Artifact, ArtifactStep

from experiments.evals.served_lm_eval import (
    ServedLmEvalBenchmark,
    brokered_lm_eval_configs,
    brokered_lm_eval_step,
    served_lm_eval_command,
)

CRUXEVAL_BENCHMARK = ServedLmEvalBenchmark(
    tasks=("cruxeval_input", "cruxeval_output"),
    output_path="/tmp/served-qwen3-cruxeval",
    job_name="served-qwen3-cruxeval",
    confirm_run_unsafe_code=True,
)


def cruxeval_step(*, version: str, limit: int | None = None) -> ArtifactStep[Artifact]:
    inference, eval_run = brokered_lm_eval_configs(
        CRUXEVAL_BENCHMARK,
        limit=limit,
        timeout_seconds=None,
        priority=0,
    )
    return brokered_lm_eval_step(
        inference,
        eval_run,
        name="evals/qwen3-0.6b/cruxeval",
        version=version,
        parent_resources=ResourceConfig.with_cpu(
            cpu=CRUXEVAL_BENCHMARK.parent_cpu,
            ram=CRUXEVAL_BENCHMARK.parent_ram,
            disk=CRUXEVAL_BENCHMARK.parent_disk,
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
