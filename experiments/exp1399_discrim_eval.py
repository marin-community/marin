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

"""Run discrimination evaluations on Gemstone models.

This experiment runs :func:`lm_evaluation_harness` on the
``discrim_eval_implicit`` and ``discrim_eval_explicit`` tasks for all Gemstone
checkpoints that have been cooled down. We re-use the Gemstone utilities from
``exp1342_gemstones_scaling_law.py`` to enumerate the models and filter for the
cooldown checkpoints at the 10% mark.
"""

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V4_8
from experiments.exp1342_gemstones_scaling_law import (
    gemstone_splits,
    roughly_equals,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main, output_path_of


def create_eval_steps() -> list:
    tasks = (
        EvalTaskConfig("discrim_eval_implicit", 0),
        EvalTaskConfig("discrim_eval_explicit", 0),
    )

    steps = []
    for config in gemstone_splits["cooldown"]:
        if roughly_equals(config.step, int(config.cooldown_start_step + (0.1 * config.cooldown_start_step))):
            try:
                model = config.model_id
                revision = config.revision
                gemstone_model = gemstone_splits["cooldown"][config]

                step = evaluate_levanter_lm_evaluation_harness(
                    model_name=f"{model}@{revision}",
                    model_path=output_path_of(gemstone_model),
                    evals=tasks,
                    resource_config=SINGLE_TPU_V4_8,
                )
                steps.append(step)
            except ValueError as e:
                print(f"Skipping {model}/{revision}: {e}")

    return steps


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    all_steps = create_eval_steps()
    for batch in chunked(all_steps, 4):
        executor_main(batch)
