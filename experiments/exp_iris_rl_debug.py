# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Minimal RL debug experiment for validating the Iris v2 orchestration.

Runs 5 training steps with Llama 3.1 8B on MATH, with small batch sizes
and short sequences to finish fast. Use this to validate the full pipeline:
coordinator → actors → trainer + rollout worker → Arrow Flight weight transfer.
"""

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

llama_3_1_8b = ModelConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    type="llama",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint="meta-llama/Llama-3.1-8B-Instruct",
    config_class=LlamaConfig,
)


def create_debug_curriculum(run_id: str, experiment_config: RLExperimentConfig) -> CurriculumConfig:
    """Single-lesson math curriculum for debugging."""
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
        eval_frequency=5,  # eval once at the end
        micro_eval_frequency=9999999,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=10,  # tiny eval for speed
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
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
        experiment_name_suffix="iris-debug",
        # Small for fast iteration
        num_train_steps=5,
        train_batch_size=64,
        per_device_parallelism=16,
        learning_rate=2e-6,
        max_input_tokens=512,
        max_output_tokens=512,
        n_prompts=4,
        n_generations_per_prompt=16,
        # Single worker for debugging
        num_rollout_workers=1,
        train_tpu_type="v5p-8",
        inference_tpu_type="v5p-8",
        # Fast weight sync
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=300,
        inflight_weight_updates=False,
        # Wandb
        project_name="marin_iris_rl_debug",
        tags=["rl", "iris-debug", "math"],
    )

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"iris-rl-debug-{datestamp}"

    curriculum = create_debug_curriculum(name, debug_config)

    step = make_rl_step(
        name=name,
        config=debug_config,
        curriculum=curriculum,
    )

    executor_main(
        steps=[step],
        description="Iris RL debug: 5 steps, 1 worker, small batches",
    )


if __name__ == "__main__":
    main()
