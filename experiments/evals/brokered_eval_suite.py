# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.lm_eval import LmEvalResults, LmEvalRun
from marin.execution.lazy import ArtifactStep
from marin.inference.vllm import BrokeredVllmSystemConfig

from experiments.evals.served_lm_eval import brokered_lm_eval_step

BROKERED_EVAL_TASKS = (
    "cruxeval_input",
    "cruxeval_output",
    "humaneval",
)


def brokered_eval_suite(
    inference: BrokeredVllmSystemConfig,
    *,
    model_name: str,
    version: str,
    limit: int | None = None,
) -> ArtifactStep[LmEvalResults]:
    return brokered_lm_eval_step(
        inference,
        LmEvalRun(
            tasks=BROKERED_EVAL_TASKS,
            limit=limit,
            confirm_run_unsafe_code=True,
        ),
        name=f"evals/{model_name}/suite",
        version=version,
        parent_env_vars={"HF_ALLOW_CODE_EVAL": "1"},
    )
