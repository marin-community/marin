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
Example of continual pretraining under the two stage framework for Basque data (based on two_stage_config.py).
Simply load a pretrained model and do a single stage of training with choice of replay data.
"""

from experiments.two_stage.two_stage_config import TwoStageConfig, two_stage_train_step
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

BASQUE_TASKS = (EvalTaskConfig("xcopa_eu", num_fewshot=0, task_alias="xcopa_eu"),)
MATH_TASKS = (EvalTaskConfig(name="mathqa", num_fewshot=8),)

TASKS = {
    "latxa": BASQUE_TASKS,
    "octo": MATH_TASKS,
}

HF_MODELS = {
    "l3b": "meta-llama/Llama-3.2-3B",
    "l8b": "meta-llama/Meta-Llama-3.1-8B",
}

if __name__ == "__main__":
    # NUM_RARE_STEPS = 400.0  # 200M tokens
    NUM_RARE_STEPS = 4000.0  # 2B tokens
    train_steps = [
        two_stage_train_step(
            TwoStageConfig(
                rare_data_name=rare_data_name,
                common_data_name=common_data_name,
                rare_fraction=1 - replay_ratio,
                rare_stage2_allocation=1.0,
                stage2_duration=1.0,
                rare_data_epochs=rare_data_epochs,
                num_train_steps=NUM_RARE_STEPS / (1 - replay_ratio),
                lr_schedule=lr_schedule,
                lr=lr,
                lr_cooldown_duration=lr_cooldown_duration,
                train_batch_size=128,
                wandb_project_name="suhas-two-stage",
                wandb_additional_tags=["cpt", f"{rare_data_name}-{common_data_name}-cpt"],
                model_name=model_name,
                initialize_from_hf=HF_MODELS[model_name],
                # eval_harness_tasks=TASKS[rare_data_name],
                tpu_type="v4-64",
                nametag="v2"
            )
        )
        for lr_schedule, lr_cooldown_duration in [
            ("cosine", 1.0),
        ]
        for lr in [1e-5]
        for replay_ratio in [0.0, 0.5, 0.75, 0.9]
        for rare_data_name in ["octo"]
        for common_data_name in ["spj_full"]
        for rare_data_epochs in [1]
        for model_name in ["l8b"]
    ]

    executor_main(
        steps=train_steps,
    )