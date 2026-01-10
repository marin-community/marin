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

import datetime
import logging
import os

from levanter.models.qwen import QwenConfig
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_losses import RLOOLoss

from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    make_rl_step,
)

logger = logging.getLogger(__name__)


# Qwen2.5-7B-Instruct-1M for CodeR1 training
qwen25_7b_instruct_1m = ModelConfig(
    name="Qwen/Qwen2.5-7B-Instruct-1M",
    type="qwen",
    tokenizer="Qwen/Qwen2.5-7B-Instruct-1M",
    checkpoint="Qwen/Qwen2.5-7B-Instruct-1M",
    config_class=QwenConfig,  # Qwen2.5 uses Qwen2Config (not Qwen3Config)
    pip_dependency_groups=["vllm", "math"],  # Assuming we need math group too or just vllm?
)


def create_code_curriculum(run_id: str, experiment_config: RLExperimentConfig) -> CurriculumConfig:
    """Create CodeR1 curriculum for code generation training.

    Uses LeetCode 2K dataset for training and HumanEvalPlus for evaluation.
    Based on Code-R1: https://github.com/ganler/code-r1
    """

    # Code-R1 sampling params from main_grpo.sh
    code_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=experiment_config.n_prompts,
        n_generations_per_prompt=experiment_config.n_generations_per_prompt,
        max_output_tokens=experiment_config.max_output_tokens,
        top_k=4096,
        stop_tokens=None,
    )

    lessons = {
        "code_r1": LessonConfig(
            lesson_id="code_r1",
            env_config=EnvConfig(
                env_class="marin.rl.environments.code_r1_env.CodeR1Env",
                env_args={"seed": 42},
            ),
            dependencies=[],
            sampling_params=code_sampling,
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=1,  # Run HumanEvalPlus eval after every step
        micro_eval_frequency=9999999,  # Disable micro-eval
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=164,  # HumanEvalPlus has 164 problems
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    # CodeR1 training config (from Code-R1 main_grpo.sh)
    qwen25_code = RLExperimentConfig(
        model_config=qwen25_7b_instruct_1m,
        rl_loss=RLOOLoss(
            kl_coef=0.001,  # Code-R1 uses kl_loss_coef=0.001
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.28,
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,
            vocab_tile_size=152064,  # Qwen vocab size
        ),
        experiment_name_suffix="code-r1-lr=5e-7",
        train_batch_size=256,  # 16 prompts * 16 generations = 256
        per_device_parallelism=16,
        learning_rate=5e-7,  # Code-R1: actor_rollout_ref.actor.optim.lr=5e-7
        max_input_tokens=2048,  # Code-R1: data.max_prompt_length=2048
        max_output_tokens=4096,  # Code-R1: data.max_response_length=4096
        n_prompts=16,  # Code-R1: ROLLOUT_N_QUERY=16
        n_generations_per_prompt=16,  # Code-R1: ROLLOUT_N_SAMPLE=16
        inflight_weight_updates=True,
        max_rollout_step_delay=1,
    )

    experiment_configs = [qwen25_code]
    experiments = []
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for experiment_config in experiment_configs:
        model_base_name = experiment_config.model_config.name.split("/")[-1].lower()
        model_base_name = model_base_name.replace("-instruct", "i")

        # Always include timestamp to avoid cache collisions between runs
        name = f"{model_base_name}-{experiment_config.experiment_name_suffix}-{datestamp}"

        curriculum = create_code_curriculum(name, experiment_config)

        experiments.append(
            make_rl_step(
                name=name,
                config=experiment_config,
                curriculum=curriculum,
            ),
        )

    executor_main(
        steps=experiments,
        description="Async RL code training experiments",
    )


if __name__ == "__main__":
    main()
