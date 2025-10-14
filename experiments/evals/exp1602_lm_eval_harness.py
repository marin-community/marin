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
Comprehensive LM Evaluation Harness Testing
Reference: https://github.com/EleutherAI/lm-evaluation-harness
"""

from experiments.evals.evals import default_eval
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from experiments.evals.task_configs import REASONING_TASKS, EMOTIONAL_ETHICS_TASKS, LANGUAGE_TASKS, CODE_TASKS, MEDICAL_TASKS, KNOWLEDGE_TASKS, BIAS_SAFETY_TASKS, LONG_CONTEXT_TASKS, ACTION_TASKS, TRUTHFULNESS_TASKS, SPECIALIZED_TASKS
from experiments.models import qwen3_32b
from marin.execution.executor import executor_main
from experiments.models import download_model_step, ModelConfig

from experiments.tootsie.exp1529_32b_mantis_cooldown import tootsie_32b_cooldown_mantis as marin_32b

# List of models to evaluate
MODELS_TO_EVALUATE = [
    marin_32b,
    qwen3_32b,
]

# Task configurations to run
TASK_CONFIGS = REASONING_TASKS + EMOTIONAL_ETHICS_TASKS + LANGUAGE_TASKS + CODE_TASKS + MEDICAL_TASKS + KNOWLEDGE_TASKS + BIAS_SAFETY_TASKS + ACTION_TASKS + TRUTHFULNESS_TASKS + SPECIALIZED_TASKS

if __name__ == "__main__":
    # Comprehensive evaluation suite for multiple models
    eval_steps = []

    for model in MODELS_TO_EVALUATE:
        eval_steps.append(
            default_eval(
                step=model,
                resource_config=SINGLE_TPU_V5p_8_FULL,
                evals=TASK_CONFIGS,
                discover_latest_checkpoint=False,
            )
        )
    executor_main(steps=eval_steps)