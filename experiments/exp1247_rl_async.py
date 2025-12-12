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

import dataclasses
import datetime
import logging
import os

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import (
    ExecutorStep,
    OutputName,
    executor_main,
)
from marin.rl.curriculum import CurriculumConfig, LessonConfig, LessonDependency
from marin.rl.environments import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    name: str
    type: str
    tokenizer: str
    checkpoint: str
    config_class: type[HFCompatConfig]

    @property
    def safe_name(self) -> str:
        return self.name.replace("/", "-").lower()


qwen4b = ModelConfig(
    name="Qwen/Qwen3-4B-Instruct-2507",
    type="qwen",
    tokenizer="Qwen/Qwen3-4B-Instruct-2507",
    checkpoint="Qwen/Qwen3-4B-Instruct-2507",
    config_class=Qwen3Config,
)
llama1b = ModelConfig(
    name="meta-llama/Llama-3.2-1B-Instruct",
    type="llama",
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
    checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    config_class=LlamaConfig,
)
MODEL = llama1b
WANDB_PROJECT = f"rl_testing_{MODEL.name.split('/')[-1].lower()}"
MAX_TOKENS = 512
RUN_ID = f"test-{MODEL.name.split('/')[-1]}-curriculum"


def stop_tokens(tokenizer_name: str):
    """Infer the stop tokens from the given tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [tokenizer.eos_token_id]


def create_math_curriculum(run_id: str) -> CurriculumConfig:
    """Create progressive math curriculum: comparison -> easy -> medium -> hard."""
    from marin.rl.curriculum import SamplingParams

    # Default sampling params for all lessons
    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=8,
        n_generations_per_prompt=8,
        max_tokens=MAX_TOKENS,
        stop_tokens=stop_tokens(MODEL.tokenizer),
    )

    lessons = {
        "number_comparison": LessonConfig(
            lesson_id="number_comparison",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "number_comparison", "seed": 42},
            ),
            dependencies=[],
            sampling_params=default_sampling,
        ),
        "addition_easy": LessonConfig(
            lesson_id="addition_easy",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "addition", "difficulty": "easy", "seed": 42},
            ),
            dependencies=[LessonDependency(dependency_id="number_comparison", reward_threshold=0.8)],
            sampling_params=default_sampling,
        ),
        "addition_medium": LessonConfig(
            lesson_id="addition_medium",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "addition", "difficulty": "medium", "seed": 42},
            ),
            dependencies=[LessonDependency(dependency_id="addition_easy", reward_threshold=0.8)],
            sampling_params=default_sampling,
        ),
        "addition_hard": LessonConfig(
            lesson_id="addition_hard",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "addition", "difficulty": "hard", "seed": 42},
            ),
            dependencies=[LessonDependency(dependency_id="addition_medium", reward_threshold=0.8)],
            sampling_params=default_sampling,
        ),
        "math_full": LessonConfig(
            lesson_id="math_full",
            env_config=EnvConfig(
                env_class="marin.rl.environments.math_env.MathEnv",
                env_args={},
            ),
            dependencies=[LessonDependency(dependency_id="addition_medium", reward_threshold=0.8)],
            sampling_params=default_sampling,
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=100,
        actor_name=f"curriculum-{run_id}",
    )


def rl_train(name: str) -> ExecutorStep:
    hf_config = AutoConfig.from_pretrained(MODEL.name)
    config = MODEL.config_class.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    model_config = dataclasses.replace(config, max_seq_len=MAX_TOKENS, tokenizer=MODEL.tokenizer)

    _ = WandbConfig

    trainer_config = TrainerConfig(
        # wandb is persistently crashing
        tracker=WandbConfig(
            project="rl-mockenv-testing",
            name=name,
            tags=["rl", "math", MODEL.name.split("/")[-1]],
        ),
        # tracker=TensorboardConfig(
        #     logdir=OutputName("tblogs"),
        # ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        # Set the train batch size to num_rollout_workers * n_generations * n_prompts
        # to ensure we accept an entire training batch from the rollout workers.
        train_batch_size=64 * 4,
        # microbatch to avoid OOM
        per_device_parallelism=16,
        num_train_steps=50000,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        ray=RayConfig(auto_start_cluster=False),
    )

    opt_config = AdamConfig(
        learning_rate=1e-7,
        weight_decay=1e-2,
        warmup=100,
        lr_schedule="constant",
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=OutputName("rollouts"),
    )

    curriculum_config = create_math_curriculum(name)

    config = RLJobConfig(
        inference_type="vllm",
        model=model_config,
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=opt_config,
            rl_loss=RLOOLoss(kl_coef=0.01, clip_epsilon=0.2),
            replay_buffer=ReplayBufferConfig(
                capacity=4096,
                alpha=3,
                max_samples=1,
                max_rollout_step_delay=1,
            ),
        ),
        curriculum=curriculum_config,
        tokenizer=MODEL.tokenizer,
        initial_checkpoint=MODEL.checkpoint,
        rollout_storage=rollout_storage,
        run_id=name,
        log_freq=10,
        run_config=RunConfig(
            train_tpu_type="v5litepod-4",
            num_train_slices=1,
            num_rollout_workers=1,
            inference_tpu_type="v5litepod-4",
        ),
    )

    # Enable synchronous (on-policy) training mode for testing
    config = config.with_on_policy_training()

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=RLJob.make_step_fn(),
        config=config,
        pip_dependency_groups=["post_training"],
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    experiments = [
        rl_train(name=f"{MODEL.safe_name}-math-rl-test-power-{datestamp}"),
    ]

    executor_main(
        steps=experiments,
        description="Async RL math training experiments",
    )


if __name__ == "__main__":
    main()
