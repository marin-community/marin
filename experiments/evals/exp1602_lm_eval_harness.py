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
from experiments.evals.task_configs import *
from experiments.models import llama_3_1_8b, marin_8b_base, olmo_2_base_8b, qwen3_0_6b
from marin.execution.executor import executor_main
from experiments.models import download_model_step, ModelConfig

# from experiments.tootsie.exp1529_32b_mantis_cooldown import tootsie_32b_cooldown_mantis as marin_32b


marin_32b = download_model_step(ModelConfig(hf_repo_id="tootsie/tootsie-32b-cooldown-bison-adamc", hf_revision="step-192000"))
qwen3_32b = download_model_step(ModelConfig(hf_repo_id="Qwen/Qwen3-32B", hf_revision="9216db5781bf21249d130ec9da846c4624c16137"))
qwen2_5_32b = download_model_step(ModelConfig(hf_repo_id="Qwen/Qwen2.5-32B", hf_revision="1818d35814b8319459f4bd55ed1ac8709630f003"))
olmo_32b = download_model_step(ModelConfig(hf_repo_id="allenai/OLMo-2-0325-32B", hf_revision="stage2-ingredient3-step9000-tokens76B"))
# List of models to evaluate
MODELS_TO_EVALUATE = [
    marin_32b,
    # olmo_32b, # not supported in lm-eval-harness?
    # qwen3_32b,
    # qwen2_5_32b, # NotImplementedError: generate_until requires a model with paged decode support (initial_cache/decode).
    # qwen3_0_6b,
]

# Task configurations to run
TASK_CONFIGS = [("all", REASONING_TASKS + EMOTIONAL_ETHICS_TASKS + LANGUAGE_TASKS + CODE_TASKS + MEDICAL_TASKS + KNOWLEDGE_TASKS + BIAS_SAFETY_TASKS + LONG_CONTEXT_TASKS + ACTION_TASKS + TRUTHFULNESS_TASKS + SPECIALIZED_TASKS)]
# TASK_CONFIGS = [
#     ("reasoning", REASONING_TASKS),
#     # ("math", MATH_TASKS),
#     ("emotional_ethics", EMOTIONAL_ETHICS_TASKS),
#     ("language", LANGUAGE_TASKS),
#     ("code", CODE_TASKS),
#     ("medical", MEDICAL_TASKS),
#     ("knowledge", KNOWLEDGE_TASKS),
#     ("bias_safety", BIAS_SAFETY_TASKS),
#     ("long_context", LONG_CONTEXT_TASKS),
#     ("action", ACTION_TASKS),
#     ("truthfulness", TRUTHFULNESS_TASKS),
#     ("specialized", SPECIALIZED_TASKS),
# ]
# TASK_CONFIGS = [("language", LANGUAGE_TASKS), 
# ("code", CODE_TASKS),
# ("medical", MEDICAL_TASKS),
# ("knowledge", KNOWLEDGE_TASKS)]
# TASK_CONFIGS = [("bias_safety", BIAS_SAFETY_TASKS),
# ("long_context", LONG_CONTEXT_TASKS),
# ("action", ACTION_TASKS),
# ("truthfulness", TRUTHFULNESS_TASKS),
# ("specialized", SPECIALIZED_TASKS)]


if __name__ == "__main__":
    # Comprehensive evaluation suite for multiple models
    eval_steps = []

    for model in MODELS_TO_EVALUATE:
        for task_name, task_config in TASK_CONFIGS:
            eval_steps.append(
                default_eval(
                    step=model,
                    resource_config=SINGLE_TPU_V5p_8_FULL,
                    evals=task_config,
                    max_eval_instances=1000,
                    discover_latest_checkpoint=False,
                )
            )
    executor_main(steps=eval_steps)
