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
Evaluate the multilingual CPT continuation (exp1457) and Llama 3.1 8B on multilingual LM Eval Harness tasks.
"""

from collections.abc import Iterable
from dataclasses import replace

from experiments.evals.engine_configs import DEFAULT_LM_EVAL_MODEL_KWARGS
from experiments.evals.evals import default_eval, evaluate_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8
from experiments.evals.task_configs import MULTILINGUAL_LM_EVAL_LOGPROB_TASKS
from experiments.multilingual.exp1457_multilingual_cpt import multilingual_cpt_8b_fineweb2_hq
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main


def _create_per_task_eval_steps(tasks: Iterable[EvalTaskConfig]) -> list[ExecutorStep]:
    """Return one evaluation step per LM Eval Harness task for the multilingual CPT model."""

    per_task_steps: list[ExecutorStep] = []
    for task in tasks:
        eval_step = default_eval(
            step=multilingual_cpt_8b_fineweb2_hq,
            resource_config=SINGLE_TPU_V5p_8,
            evals=(task,),
        )
        task_label = task.task_alias or task.name
        per_task_steps.append(replace(eval_step, name=f"{eval_step.name}/{task_label}"))

    return per_task_steps


def _create_llama3_per_task_eval_steps(tasks: Iterable[EvalTaskConfig]) -> list[ExecutorStep]:
    """Return one evaluation step per LM Eval Harness task for Llama 3.1 8B."""

    llama3_steps: list[ExecutorStep] = []
    for task in tasks:
        eval_step = evaluate_lm_evaluation_harness(
            model_name="llama-3.1-8b-base",
            model_path="meta-llama/Meta-Llama-3.1-8B",
            evals=(task,),
            max_eval_instances=None,
            engine_kwargs=DEFAULT_LM_EVAL_MODEL_KWARGS,
            resource_config=SINGLE_TPU_V5p_8,
            apply_chat_template=False,
            discover_latest_checkpoint=False,
        )
        task_label = task.task_alias or task.name
        llama3_steps.append(replace(eval_step, name=f"{eval_step.name}/{task_label}"))

    return llama3_steps


multilingual_eval_steps = [
    # *_create_per_task_eval_steps(MULTILINGUAL_LM_EVAL_LOGPROB_TASKS),
    *_create_llama3_per_task_eval_steps(MULTILINGUAL_LM_EVAL_LOGPROB_TASKS),
]


if __name__ == "__main__":
    for i in range(0, len(multilingual_eval_steps), 4):
        batch = multilingual_eval_steps[i : i + 4]
        executor_main(steps=batch)
