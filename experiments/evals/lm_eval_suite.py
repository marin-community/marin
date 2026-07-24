# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lm-eval task suite used by served-model experiments."""

from marin.evaluation.lm_eval import LmEvalResults, LmEvalRun
from marin.evaluation.serving_config import InferenceLaunch
from marin.execution.lazy import ArtifactStep
from marin.experiment.evaluation import lm_eval_step

LM_EVAL_TASKS = (
    "cruxeval_input",
    "cruxeval_output",
    "humaneval",
)


def lm_eval_suite(
    inference: InferenceLaunch,
    *,
    model_name: str,
    version: str,
    limit: int | None = None,
) -> ArtifactStep[LmEvalResults]:
    """Evaluate the served model on the repository's code task menu."""
    return lm_eval_step(
        inference,
        LmEvalRun(
            tasks=LM_EVAL_TASKS,
            limit=limit,
            confirm_run_unsafe_code=True,
        ),
        name=f"evals/{model_name}/suite",
        version=version,
        parent_env_vars={"HF_ALLOW_CODE_EVAL": "1"},
    )
