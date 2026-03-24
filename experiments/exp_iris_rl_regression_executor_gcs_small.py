# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression probe E3: executor + GCS + small RL launch.

This changes only topology relative to the green direct GCS probe:
- executor_main wrapper
- executor-managed RL step
- regional GCS-backed model artifact
- tiny 5-step envelope
"""

import datetime
import logging
import os

from experiments.models import llama_3_1_8b_instruct
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    config_class_path,
    executor_main_config_for_rl_experiment,
    make_rl_step,
)
from marin.rl.rl_losses import RLOOLoss

logger = logging.getLogger(__name__)

llama_3_1_8b = ModelConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    type="llama",
    artifact=llama_3_1_8b_instruct,
    config_class_path=config_class_path(LlamaConfig),
)


def create_debug_curriculum(run_id: str, experiment_config: RLExperimentConfig) -> CurriculumConfig:
    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=experiment_config.n_prompts,
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
        eval_frequency=5,
        micro_eval_frequency=9999999,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=10,
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def main() -> None:
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    debug_config = RLExperimentConfig(
        model_config=llama_3_1_8b,
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
        experiment_name_suffix="exec-gcs-small",
        project_name="marin_iris_rl_debug",
        tags=["rl", "iris-debug", "regression", "e3", "exec-gcs-small"],
        num_train_steps=5,
        train_batch_size=64,
        per_device_parallelism=16,
        learning_rate=2e-6,
        max_input_tokens=512,
        max_output_tokens=512,
        n_prompts=4,
        n_generations_per_prompt=16,
        num_rollout_workers=1,
        train_tpu_type="v5p-8",
        inference_tpu_type="v5p-8",
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=300,
        inflight_weight_updates=False,
    )

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"exec-gcs-small-{datestamp}"
    curriculum = create_debug_curriculum(name, debug_config)
    step = make_rl_step(
        name=name,
        config=debug_config,
        curriculum=curriculum,
    )

    executor_main(
        executor_main_config_for_rl_experiment(debug_config),
        steps=[step],
        description="Iris RL regression probe E3: executor + GCS + small",
    )


if __name__ == "__main__":
    main()
