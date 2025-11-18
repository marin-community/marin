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
#TBD: Math Scaling

Evaluates models log-likelihood on math reasoning traces.
"""

import os
import logging
from dataclasses import dataclass
from functools import lru_cache


from experiments.llama import llama3_tokenizer
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from marin.execution.executor import executor_main, ExecutorStep, output_path_of, this_output_path
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL, SINGLE_TPU_V4_8, TPU_V6E_8_STRICT_PACK
from experiments.models import ModelConfig as HFModelConfig, download_model_step

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import EvalTaskConfig
# from experiments.evals.task_configs import (
#     REASONING_TASKS,
#     MATH_TASKS,
# )

import logging
logging.getLogger("levanter.eval_harness").setLevel(logging.WARNING)

# Mathematical and Arithmetic Tasks
MATH_TASKS = (
    EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot"),  # included in core tasks
    # EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
    # EvalTaskConfig("arithmetic_1dc", 0, task_alias="arithmetic_1dc_0shot"),
    # EvalTaskConfig("arithmetic_2da", 0, task_alias="arithmetic_2da_0shot"),
    # EvalTaskConfig("arithmetic_2dm", 0, task_alias="arithmetic_2dm_0shot"),
    # EvalTaskConfig("arithmetic_2ds", 0, task_alias="arithmetic_2ds_0shot"),
    # EvalTaskConfig("arithmetic_3da", 0, task_alias="arithmetic_3da_0shot"),
    # EvalTaskConfig("arithmetic_3ds", 0, task_alias="arithmetic_3ds_0shot"),
    # EvalTaskConfig("arithmetic_4da", 0, task_alias="arithmetic_4da_0shot"),
    # EvalTaskConfig("arithmetic_4ds", 0, task_alias="arithmetic_4ds_0shot"),
    # EvalTaskConfig("arithmetic_5da", 0, task_alias="arithmetic_5da_0shot"),
    # EvalTaskConfig("arithmetic_5ds", 0, task_alias="arithmetic_5ds_0shot"),
    # EvalTaskConfig("asdiv", 0, task_alias="asdiv_0shot"),
    # EvalTaskConfig("hendrycks_math_algebra", 0, task_alias="hendrycks_math_algebra_0shot"),
    # EvalTaskConfig("hendrycks_math_counting_and_prob", 0, task_alias="hendrycks_math_counting_and_prob_0shot"),
    # EvalTaskConfig("hendrycks_math_geometry", 0, task_alias="hendrycks_math_geometry_0shot"),
    # EvalTaskConfig("hendrycks_math_intermediate_algebra", 0, task_alias="hendrycks_math_intermediate_algebra_0shot"),
    # EvalTaskConfig("hendrycks_math_num_theory", 0, task_alias="hendrycks_math_num_theory_0shot"),
    # EvalTaskConfig("hendrycks_math_prealgebra", 0, task_alias="hendrycks_math_prealgebra_0shot"),
    # EvalTaskConfig("hendrycks_math_precalc", 0, task_alias="hendrycks_math_precalc_0shot"),
    # EvalTaskConfig("mathqa", 0, task_alias="mathqa_0shot"),
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name: str
    revision: str
    tokenizer: str | None = None  # Optional: if None, uses model_name as tokenizer


def get_directory_friendly_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(".", "-")


def truncate_model_name(model_name: str, max_length: int = 62) -> str:
    """Truncate model name to max_length if it exceeds that length."""
    return model_name[:max_length] if len(model_name) > max_length else model_name


def build_steps(models: list[ModelConfig], tasks: tuple[EvalTaskConfig]) -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for model_config in models:
        # Download model and load config dynamically from HuggingFace
        model_instance = download_model_step(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )

        steps.append(
            default_eval(
                step=model_instance,
                resource_config=SINGLE_TPU_V5p_8_FULL,
                evals=tasks,
                discover_latest_checkpoint=False,
            )
        )


    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return
    
    # tasks = (
    #     REASONING_TASKS
    #     + MATH_TASKS
    # )
    tasks = (MATH_TASKS)
    
    models = [
        # ModelConfig(model_name="marin-community/marin-8b-base", revision="main", tokenizer=llama3_tokenizer),
        # ModelConfig(model_name="allenai/OLMo-2-0325-32B", revision="main"),
        # ModelConfig(model_name="Qwen/Qwen3-32B", revision="main"),
        # ModelConfig(model_name="meta-llama/Llama-3.1-8B", revision="main", tokenizer=llama3_tokenizer),
        # ModelConfig(model_name="meta-llama/Llama-3.2-1B", revision="main", tokenizer=llama3_tokenizer),
        # ModelConfig(model_name="allenai/OLMo-2-1124-7B", revision="main"),
        ModelConfig(model_name="Qwen/Qwen3-0.6B", revision="main"),
        # ModelConfig(model_name="Qwen/Qwen3-1.7B", revision="main"),
        # ModelConfig(model_name="Qwen/Qwen3-4B", revision="main"),
        # ModelConfig(model_name="Qwen/Qwen3-8B", revision="main"),
        # ModelConfig(model_name="Qwen/Qwen3-0.6B-Base", revision="main", tokenizer="Qwen/Qwen3-0.6B"),
        # ModelConfig(model_name="Qwen/Qwen3-1.7B-Base", revision="main", tokenizer="Qwen/Qwen3-1.7B"),
        # ModelConfig(model_name="Qwen/Qwen3-4B-Base", revision="main", tokenizer="Qwen/Qwen3-4B"),
        # ModelConfig(model_name="Qwen/Qwen3-8B-Base", revision="main", tokenizer="Qwen/Qwen3-8B"),
    ]
    
    executor_main(steps=build_steps(models=models, tasks=tasks), description="Math Reasoning - Log Probability Evaluation")


if __name__ == "__main__":
    main()
