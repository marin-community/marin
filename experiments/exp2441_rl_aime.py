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

"""DeepMath-Zero-7B replication in Marin.

This experiment replicates DeepMath-Zero-7B using RL training on AIME-style
math problems. Uses zwhe99/DeepMath-103K for training data and
zwhe99/Qwen2.5-7B-orz as the base model.

Hyperparameters follow DeepMath/scripts/train/deepmath-zero-7b.sh:
- GRPO advantage estimator
- lr=1e-6, weight_decay=0.1, grad_clip=1.0
- train_batch_size=512, max_prompt_length=2048, max_response_length=10240
- n=16 generations per prompt, temperature=1.0
- clip_ratio_low=0.2, clip_ratio_high=0.28
- token-level loss, no KL penalty
- overlong_buffer: len=2048, penalty_factor=1.0
- filter_groups by accuracy metric

Reference: https://github.com/zwhe99/DeepMath
"""
# nodryrun because vLLM is not installed by default

import datetime
import logging
import os

from levanter.models.qwen import QwenConfig
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_losses import RLOOLoss

from experiments.models import (
    ModelConfig as HFModelConfig,
    levanter_model_step,
)

from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    make_rl_step,
)

logger = logging.getLogger(__name__)

# DeepMath-Zero-7B uses zwhe99/Qwen2.5-7B-orz as the base model
# (a Qwen2.5-7B variant with the "orz" chat template for think/answer format)
qwen2_5_7b_orz_hf_config = HFModelConfig(
    hf_repo_id="zwhe99/Qwen2.5-7B-orz",
    hf_revision="5625e85",
    config_class=QwenConfig,
)

qwen_2_5_7b_orz = ModelConfig(
    name="zwhe99/Qwen2.5-7B-orz",
    type="qwen",
    tokenizer="zwhe99/Qwen2.5-7B-orz",
    checkpoint=levanter_model_step(qwen2_5_7b_orz_hf_config).as_input_name(),
    config_class=QwenConfig,
)


def create_aime_curriculum(run_id: str, experiment_config: RLExperimentConfig) -> CurriculumConfig:
    """Create AIME curriculum using DeepMath-103K dataset.

    Uses zwhe99/DeepMath-103K for training data following
    the DeepMath-Zero approach for AIME-style math problems.
    """
    # DeepMath training sampling parameters:
    # temperature=1.0, top_p=1.0, n=16
    default_sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        n_prompts=experiment_config.n_prompts,
        n_generations_per_prompt=experiment_config.n_generations_per_prompt,
        max_output_tokens=experiment_config.max_output_tokens,
        top_k=4096,
        stop_tokens=None,
    )
    # DeepMath evaluation sampling parameters:
    # temperature=0.6, top_p=0.95, n=16
    eval_sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        n_prompts=experiment_config.n_prompts,
        n_generations_per_prompt=experiment_config.n_generations_per_prompt,
        max_output_tokens=experiment_config.max_output_tokens,
        top_k=4096,
        stop_tokens=None,
    )

    lessons = {
        "deepmath_103k": LessonConfig(
            lesson_id="deepmath_103k",
            env_config=EnvConfig(
                env_class="marin.rl.environments.aime_env.AimeEnv",
                env_args={
                    "seed": 42,
                    "train_dataset_name": "zwhe99/DeepMath-103K",
                    "eval_dataset_name": "math-ai/aime25",
                    "overlong_buffer_len": 2048,
                    "overlong_penalty_factor": 1.0,
                },
            ),
            dependencies=[],
            sampling_params=default_sampling,
            eval_sampling_params=eval_sampling,
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=1,
        micro_eval_frequency=9999999,  # Effectively disable micro-eval
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=30,  # AIME25 has 30 problems
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    # DeepMath-Zero-7B hyperparameters from deepmath-zero-7b.sh
    # System prompt from DeepMath ("simplerl" template):
    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    deepmath_zero_7b = RLExperimentConfig(
        model_config=qwen_2_5_7b_orz,
        system_prompt=SYSTEM_PROMPT,
        pip_dependency_groups=["vllm", "aime"],
        run_env_vars={"LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=65536"},
        rl_loss=RLOOLoss(
            kl_coef=0.0,  # No KL penalty (use_kl_loss=False, kl_loss_coef=0.0)
            clip_epsilon_low=0.2,  # clip_ratio_low=0.2
            clip_epsilon_high=0.28,  # clip_ratio_high=0.28
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,  # filter_groups.enable=True
            vocab_tile_size=32064,
        ),
        experiment_name_suffix="deepmath-zero-7b",
        # data.train_batch_size=512
        train_batch_size=512,
        per_device_parallelism=8,
        # actor_rollout_ref.actor.optim.lr=1e-6
        learning_rate=1e-6,
        # actor_rollout_ref.actor.optim.weight_decay=0.1
        weight_decay=0.1,
        # actor_rollout_ref.actor.optim.lr_warmup_steps=10
        warmup=10,
        # actor_rollout_ref.actor.grad_clip=1.0
        max_grad_norm=1.0,
        # data.max_prompt_length=2048
        max_input_tokens=2048,
        # data.max_response_length=10240
        max_output_tokens=10240,
        # Sampling: n=16 generations per prompt
        # data.gen_batch_size=1536 / n=16 => ~96 prompts per generation batch
        n_prompts=96,
        n_generations_per_prompt=16,
        # trainer.total_training_steps=500
        num_train_steps=500,
        inflight_weight_updates=True,
        max_rollout_step_delay=1,
        # Tags for tracking
        tags=["rl", "aime", "deepmath", "qwen"],
    )

    experiment_configs = [deepmath_zero_7b]
    experiments = []
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for experiment_config in experiment_configs:
        model_base_name = experiment_config.model_config.name.split("/")[-1].lower()

        # Always include timestamp to avoid cache collisions between runs
        name = f"{model_base_name}-{experiment_config.experiment_name_suffix}-{datestamp}"

        curriculum = create_aime_curriculum(name, experiment_config)

        experiments.append(
            make_rl_step(
                name=name,
                config=experiment_config,
                curriculum=curriculum,
            ),
        )

    executor_main(
        steps=experiments,
        description="DeepMath-Zero-7B replication: RL math training with Qwen2.5-7B-orz",
    )


if __name__ == "__main__":
    main()
