# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the selected LM Eval Harness tasks across a set of Marin, Qwen 2.5, OLMo 2, Llama 3, and OLMo 3 models.
"""

from collections.abc import Iterable
from dataclasses import replace

from fray.cluster import ResourceConfig
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import LM_EVAL_HARNESS_SELECTED_TASKS
from experiments.models import (
    llama_3_1_8b,
    llama_3_70b,
    marin_8b_base,
    marin_32b_base,
    olmo_2_base_32b,
    olmo_2_base_8b,
    olmo_3_32b,
    olmo_3_7b,
    qwen2_5_32b,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

MARIN_MODELS: tuple[ExecutorStep, ...] = (marin_8b_base, marin_32b_base)
QWEN_2_5_MODELS: tuple[ExecutorStep, ...] = (qwen2_5_32b, )
OLMO_2_MODELS: tuple[ExecutorStep, ...] = (olmo_2_base_8b, olmo_2_base_32b)
LLAMA_3_MODELS: tuple[ExecutorStep, ...] = (llama_3_1_8b, llama_3_70b)
OLMO_3_MODELS: tuple[ExecutorStep, ...] = (olmo_3_7b, olmo_3_32b)

ALL_MODEL_STEPS: tuple[ExecutorStep, ...] = (
    # *MARIN_MODELS,
    # *QWEN_2_5_MODELS,
    # *OLMO_2_MODELS,
    # *LLAMA_3_MODELS,
    *OLMO_3_MODELS,
)


def _create_per_task_eval_steps(model_step: ExecutorStep, tasks: Iterable[EvalTaskConfig]) -> list[ExecutorStep]:
    """Return one evaluation step per LM Eval Harness task for a given model."""

    per_task_steps: list[ExecutorStep] = []
    for task in tasks:
        eval_step = default_eval(
            step=model_step,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
            evals=(task,),
            discover_latest_checkpoint=False,
        )
        task_label = task.task_alias or task.name
        # Make it obvious which harness task is running to simplify scheduling/debugging.
        per_task_steps.append(replace(eval_step, name=f"{eval_step.name}/{task_label}"))

    return per_task_steps


eval_steps: list[ExecutorStep] = []
for model_step in ALL_MODEL_STEPS:
    eval_steps.extend(_create_per_task_eval_steps(model_step, LM_EVAL_HARNESS_SELECTED_TASKS))

if __name__ == "__main__":
    # executor_main(steps=eval_steps)
    for i in range(0, len(eval_steps), 4):
        executor_main(steps=eval_steps[i : i + 4])
