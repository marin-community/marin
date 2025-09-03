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

"""Run CORE evaluations on Gemstone models."""

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU, MMLU_PRO_5_SHOT, EvalTaskConfig
from experiments.exp1342_gemstones_scaling_law import (
    gemstone_splits,
    roughly_equals,
)
from experiments.models import ModelConfig, download_model_step
from marin.execution.executor import executor_main, output_path_of


def create_eval_steps() -> list:
    tasks = (
        *CORE_TASKS_PLUS_MMLU,
        MMLU_PRO_5_SHOT,
        EvalTaskConfig("commonsense_qa_sl", num_fewshot=10),
        EvalTaskConfig("mmlu_sl", num_fewshot=0, task_alias="mmlu_sl_0_shot"),
        EvalTaskConfig("mmlu_sl", num_fewshot=5, task_alias="mmlu_sl_5_shot"),
        EvalTaskConfig("mmlu_sl_verb", num_fewshot=0, task_alias="mmlu_sl_verb_0_shot"),
        EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
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
                    resource_config=SINGLE_TPU_V5p_8,
                )
                steps.append(step)
            except ValueError as e:
                print(f"Skipping {model}/{revision}: {e}")

    big_models = [
        ("allenai/OLMo-2-1124-7B", "7df9a82"),
        ("allenai/OLMo-2-1124-13B", "3fefddc"),
        ("meta-llama/Llama-3.1-8B", "d04e592"),
        ("common-pile/comma-v0.1-2t", "3fba893"),
    ]
    for model, revision in big_models:
        model_instance = download_model_step(ModelConfig(hf_repo_id=model, hf_revision=revision))

        step = evaluate_levanter_lm_evaluation_harness(
            model_name=f"{model}@{revision}",
            model_path=output_path_of(model_instance),
            evals=tasks,
            resource_config=SINGLE_TPU_V5p_8,
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    all_steps = create_eval_steps()
    executor_main(all_steps)
