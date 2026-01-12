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
# nodryrun because vLLM is not installed by default

import datetime
import logging
import os

from levanter.models.llama import LlamaConfig
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


marin_8b_instruct = ModelConfig(
    name="marin-community/marin-8b-instruct",
    type="llama",
    tokenizer="marin-community/marin-8b-instruct",
    checkpoint="marin-community/marin-8b-instruct",
    config_class=LlamaConfig,
)


def create_math_curriculum(run_id: str, experiment_config: RLExperimentConfig) -> CurriculumConfig:
    """Create progressive math curriculum: comparison -> easy -> medium -> hard."""

    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=experiment_config.n_prompts,  # Overdo it since we know there are some with no signal?
        n_generations_per_prompt=experiment_config.n_generations_per_prompt,
        max_output_tokens=experiment_config.max_output_tokens,
        top_k=4096,
        stop_tokens=None,
    )

    lessons = {
        "math_full": LessonConfig(
            lesson_id="math_full",
            env_config=EnvConfig(
                env_class="marin.rl.environments.math_env.MathEnv",
                env_args={"seed": 42},
            ),
            dependencies=[],
            sampling_params=default_sampling,
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=1,  # Run full eval after every step
        micro_eval_frequency=9999999,  # Effectively disable micro-eval
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=500,  # for math500
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    marin_8b = RLExperimentConfig(
        model_config=marin_8b_instruct,
        rl_loss=RLOOLoss(
            kl_coef=0.0,
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.28,
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,
            vocab_tile_size=32064,
        ),
        experiment_name_suffix="math-lr=2e-6-bs=1024",
        train_batch_size=1024,
        per_device_parallelism=16,
        learning_rate=2e-6,
        max_input_tokens=1024,
        max_output_tokens=1024,
        n_prompts=64,
        n_generations_per_prompt=16,
        inflight_weight_updates=True,
        max_rollout_step_delay=1,
    )

    experiment_configs = [marin_8b]
    experiments = []
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for experiment_config in experiment_configs:
        model_base_name = experiment_config.model_config.name.split("/")[-1].lower()
        model_base_name = model_base_name.replace("-instruct", "i")

        # Always include timestamp to avoid cache collisions between runs
        name = f"{model_base_name}-{experiment_config.experiment_name_suffix}-{datestamp}"

        curriculum = create_math_curriculum(name, experiment_config)

        experiments.append(
            make_rl_step(
                name=name,
                config=experiment_config,
                curriculum=curriculum,
            ),
        )

    executor_main(
        steps=experiments,
        description="Async RL math training experiments",
    )


if __name__ == "__main__":
    main()
