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
Evaluate the multilingual CPT continuation (exp1457) against multilingual LM Eval Harness tasks.
"""

from collections.abc import Iterable
from dataclasses import replace

from experiments.evals.evals import default_eval
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8
from experiments.evals.task_configs import MULTILINGUAL_LM_EVAL_LOGPROB_TASKS
from experiments.multilingual.exp1457_multilingual_cpt import multilingual_cpt_8b_fineweb2_hq
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main


def _create_per_task_eval_steps(tasks: Iterable[EvalTaskConfig]) -> list[ExecutorStep]:
    """Return one evaluation step per LM Eval Harness task."""

    per_task_steps: list[ExecutorStep] = []
    for task in tasks:
        eval_step = default_eval(
            step=multilingual_cpt_8b_fineweb2_hq,
            resource_config=SINGLE_TPU_V5p_8,
            evals=(task,),
        )
        task_label = task.task_alias or task.name
        # Make it obvious which harness task is running to simplify scheduling/debugging.
        per_task_steps.append(replace(eval_step, name=f"{eval_step.name}/{task_label}"))

    return per_task_steps


multilingual_eval_steps = _create_per_task_eval_steps(MULTILINGUAL_LM_EVAL_LOGPROB_TASKS)

# TODO: add per-task generative evals once MULTILINGUAL_LM_EVAL_GENERATIVE_TASKS is ready.

if __name__ == "__main__":
    for i in range(0, len(multilingual_eval_steps), 4):
        batch = multilingual_eval_steps[i : i + 4]
        executor_main(steps=batch)
