# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate the multilingual CPT continuation (exp1457) and baseline 7B/8B models on multilingual LM Eval Harness tasks.
"""

from collections.abc import Iterable
from dataclasses import replace

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MULTILINGUAL_LM_EVAL_LOGPROB_TASKS
from experiments.multilingual.exp1457_multilingual_cpt import multilingual_cpt_8b_fineweb2_hq
from experiments.models import apertus_8b, llama_3_1_8b, olmo_2_base_8b, olmo_3_1025_7b, qwen2_5_7b
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

SINGLE_TPU_V5p_8 = ResourceConfig.with_tpu("v5p-8")


def _create_per_task_eval_steps(
    model_step: ExecutorStep,
    tasks: Iterable[EvalTaskConfig],
    apply_chat_template: bool | None = None,
    discover_latest_checkpoint: bool | None = None,
) -> list[ExecutorStep]:
    """Return one evaluation step per LM Eval Harness task for the given model."""
    steps: list[ExecutorStep] = []
    for task in tasks:
        kwargs: dict = dict(
            step=model_step,
            resource_config=SINGLE_TPU_V5p_8,
            evals=(task,),
        )
        if apply_chat_template is not None:
            kwargs["apply_chat_template"] = apply_chat_template
        if discover_latest_checkpoint is not None:
            kwargs["discover_latest_checkpoint"] = discover_latest_checkpoint

        eval_step = default_eval(**kwargs)
        task_label = (task.task_alias or task.name).replace(" ", "_")
        steps.append(replace(eval_step, name=f"{eval_step.name}/{task_label}"))

    return steps


multilingual_eval_steps = [
    *_create_per_task_eval_steps(multilingual_cpt_8b_fineweb2_hq, MULTILINGUAL_LM_EVAL_LOGPROB_TASKS),
    *_create_per_task_eval_steps(llama_3_1_8b, MULTILINGUAL_LM_EVAL_LOGPROB_TASKS, False, False),
    *_create_per_task_eval_steps(qwen2_5_7b, MULTILINGUAL_LM_EVAL_LOGPROB_TASKS, False, False),
    *_create_per_task_eval_steps(olmo_2_base_8b, MULTILINGUAL_LM_EVAL_LOGPROB_TASKS, False, False),
    *_create_per_task_eval_steps(olmo_3_1025_7b, MULTILINGUAL_LM_EVAL_LOGPROB_TASKS, False, False),
    *_create_per_task_eval_steps(apertus_8b, MULTILINGUAL_LM_EVAL_LOGPROB_TASKS, False, False),
]


if __name__ == "__main__":
    for i in range(0, len(multilingual_eval_steps), 4):
        batch = multilingual_eval_steps[i : i + 4]
        executor_main(steps=batch)
