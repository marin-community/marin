# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Llama 3.1 8B Math500 settings for OPD smoke launchers."""

import datetime

from levanter.models.llama import LlamaConfig
from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.decoding import SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    config_class_path,
    default_train_decoding_for_experiment,
)

from experiments.models import llama_3_1_8b_instruct

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROJECT_NAME = "marin_iris_rl_debug"
DEFAULT_CHECKPOINTER_SAVE_INTERVAL = 600
DEFAULT_EVAL_FREQUENCY = 1
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_WORKER_RAM = "400g"
DEFAULT_MAX_INPUT_TOKENS = 1024
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_EVAL_N_EXAMPLES = 500
RL_STEP_VERSION = "dev"

LLAMA_3_1_8B_INSTRUCT = ModelConfig(
    name=MODEL_NAME,
    type="llama",
    artifact=llama_3_1_8b_instruct,
    config_class_path=config_class_path(LlamaConfig),
)


def build_math500_curriculum(run_id: str, config: RLExperimentConfig, eval_frequency: int) -> CurriculumConfig:
    sampling_params = SamplingParams(
        n_prompts=config.n_prompts,
        n_generations_per_prompt=config.n_generations_per_prompt,
        train_decoding=default_train_decoding_for_experiment(config),
    )

    return CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.math_env.MathEnv",
                    env_args={"seed": 42},
                ),
                dependencies=[],
                sampling_params=sampling_params,
            ),
        },
        eval_frequency=eval_frequency,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=DEFAULT_EVAL_N_EXAMPLES,
        max_seq_len=config.max_input_tokens + config.max_output_tokens,
    )


def build_run_name(config: RLExperimentConfig, explicit_run_name: str | None) -> str:
    if explicit_run_name is not None:
        return explicit_run_name

    datestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    model_base_name = config.model_config.name.split("/")[-1].lower().replace("-instruct", "i")
    return f"{model_base_name}-{config.experiment_name_suffix}-{datestamp}"
