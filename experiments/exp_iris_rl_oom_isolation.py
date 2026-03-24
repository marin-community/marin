# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OOM isolation experiment for Iris RL.

Run one controlled case at a time:

OOM_CASE=baseline_ckpt_120
OOM_CASE=no_ckpt
OOM_CASE=reduced_batch_ckpt_120
"""

import datetime
import logging
import os

from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_experiment_utils import ModelConfig, RLExperimentConfig, make_rl_step
from marin.rl.rl_losses import RLOOLoss

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

llama_3_1_8b = ModelConfig(
    name=MODEL_NAME,
    type="llama",
    tokenizer=MODEL_NAME,
    checkpoint=MODEL_NAME,
    config_class=LlamaConfig,
)


def create_math_curriculum(run_id: str, experiment_config: RLExperimentConfig) -> CurriculumConfig:
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
        eval_frequency=1,
        micro_eval_frequency=9999999,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=500,
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def build_case_config(case: str) -> RLExperimentConfig:
    base = RLExperimentConfig(
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
        experiment_name_suffix=f"oom-{case}",
        project_name="marin_iris_rl_debug",
        tags=["rl", "iris-debug", "oom-isolation", case],
        num_train_steps=24,
        train_batch_size=1024,
        per_device_parallelism=16,
        learning_rate=2e-6,
        max_input_tokens=1024,
        max_output_tokens=1024,
        n_prompts=64,
        n_generations_per_prompt=16,
        inflight_weight_updates=True,
        max_rollout_step_delay=1,
        max_weight_transfer_wait_time=0,
    )

    if case == "baseline_ckpt_120":
        base.checkpointer_save_interval = 120
        return base
    if case == "no_ckpt":
        base.checkpointer_save_interval = 86400
        return base
    if case == "reduced_batch_ckpt_120":
        base.checkpointer_save_interval = 120
        base.train_batch_size = 512
        return base

    raise ValueError(f"Unknown OOM_CASE='{case}'. Expected one of: baseline_ckpt_120, no_ckpt, reduced_batch_ckpt_120")


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping on CI.")
        return

    case = os.environ.get("OOM_CASE", "baseline_ckpt_120")
    config = build_case_config(case)
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_base_name = config.model_config.name.split("/")[-1].lower().replace("-instruct", "i")
    name = f"{model_base_name}-{config.experiment_name_suffix}-{datestamp}"
    curriculum = create_math_curriculum(name, config)

    executor_main(
        steps=[make_rl_step(name=name, config=config, curriculum=curriculum)],
        description=f"Iris RL OOM isolation case={case}",
    )


if __name__ == "__main__":
    main()
