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
from transformers import AutoConfig

from marin.execution.executor import (
    ExecutorStep,
    OutputName,
    executor_main,
)
from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.environments import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss, RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

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
qwen3_1_7b = ModelConfig(
    name="Qwen/Qwen3-1.7B",
    type="qwen",
    tokenizer="Qwen/Qwen3-1.7B",
    checkpoint="Qwen/Qwen3-1.7B",
    config_class=Qwen3Config,
)
qwen3_8b = ModelConfig(
    name="Qwen/Qwen3-8B",
    type="qwen",
    tokenizer="Qwen/Qwen3-8B",
    checkpoint="Qwen/Qwen3-8B",
    config_class=Qwen3Config,
)


qwen3_0_6b = ModelConfig(
    name="Qwen/Qwen3-0.6B",
    type="qwen",
    tokenizer="Qwen/Qwen3-0.6B",
    checkpoint="Qwen/Qwen3-0.6B",
    config_class=Qwen3Config,
)
llama_3_1_8b = ModelConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    type="llama",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint="meta-llama/Llama-3.1-8B-Instruct",
    config_class=LlamaConfig,
)
marin_8b_instruct = ModelConfig(
    name="marin-community/marin-8b-instruct",
    type="llama",
    tokenizer="marin-community/marin-8b-instruct",
    checkpoint="marin-community/marin-8b-instruct",
    config_class=LlamaConfig,
)


@dataclasses.dataclass
class ExperimentConfig:
    model_config: ModelConfig
    rl_loss: RLLossModule
    experiment_name_suffix: str

    # trainer params
    train_batch_size: int = 1024
    per_device_parallelism: int = 16

    # some sampling params
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    n_prompts: int = 64
    n_generations_per_prompt: int = 16

    debug_mode: bool = False

    inflight_weight_updates: bool = False
    """Whether to use inflight weight updates."""

    max_rollout_step_delay: int = 0
    """Maximum number of steps to delay before applying weight updates."""

    learning_rate: float = 1e-7

    max_grad_norm: float = 1.00


def get_stop_tokens(model_type: str) -> list[str]:
    """Get model-specific stop tokens."""
    if model_type == "llama":
        return ["<|eot_id|>"]
    elif model_type == "qwen":
        return ["<|im_end|>"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_math_curriculum(run_id: str, experiment_config: ExperimentConfig) -> CurriculumConfig:
    """Create progressive math curriculum: comparison -> easy -> medium -> hard."""

    # Default sampling params for all lessons
    from marin.rl.curriculum import SamplingParams

    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=experiment_config.n_prompts,  # Overdo it since we know there are some with no signal?
        n_generations_per_prompt=experiment_config.n_generations_per_prompt,
        max_output_tokens=experiment_config.max_output_tokens,
        # stop_tokens=stop_tokens(experiment_config.model_config.tokenizer),
        stop_tokens=None,
    )

    # story_sampling = SamplingParams(
    #     temperature=1.0,
    #     n_prompts=8,
    #     n_generations_per_prompt=8,
    #     max_tokens=MAX_OUTPUT_TOKENS,
    #     stop_tokens=None,
    # )

    lessons = {
        # "number_comparison": LessonConfig(
        #     lesson_id="number_comparison",
        #     env_config=EnvConfig(
        #         env_class="marin.rl.environments.mock_env.MockEnv",
        #         env_args={"task_type": "number_comparison", "seed": 42},
        #     ),
        #     dependencies=[],
        #     sampling_params=default_sampling,
        # ),
        # "addition_easy": LessonConfig(
        #     lesson_id="addition_easy",
        #     env_config=EnvConfig(
        #         env_class="marin.rl.environments.mock_env.MockEnv",
        #         env_args={"task_type": "addition", "difficulty": "easy", "seed": 42},
        #     ),
        #     dependencies=[LessonDependency(dependency_id="number_comparison", reward_threshold=0.8)],
        #     sampling_params=default_sampling,
        # ),
        # "addition_medium": LessonConfig(
        #     lesson_id="addition_medium",
        #     env_config=EnvConfig(
        #         env_class="marin.rl.environments.mock_env.MockEnv",
        #         env_args={"task_type": "addition", "difficulty": "medium", "seed": 42},
        #     ),
        #     dependencies=[LessonDependency(dependency_id="addition_easy", reward_threshold=0.8)],
        #     sampling_params=default_sampling,
        # ),
        # "addition_hard": LessonConfig(
        #     lesson_id="addition_hard",
        #     env_config=EnvConfig(
        #         env_class="marin.rl.environments.mock_env.MockEnv",
        #         env_args={"task_type": "addition", "difficulty": "hard", "seed": 42},
        #     ),
        #     dependencies=[LessonDependency(dependency_id="addition_medium", reward_threshold=0.8)],
        #     sampling_params=default_sampling,
        # ),
        "math_full": LessonConfig(
            lesson_id="math_full",
            env_config=EnvConfig(
                env_class="marin.rl.environments.math_env.MathEnv",
                env_args={"seed": 42},
            ),
            dependencies=[],
            # dependencies=[LessonDependency(dependency_id="addition_medium", reward_threshold=0.8)],
            sampling_params=default_sampling,
        ),
        # "story_generation": LessonConfig(
        #     lesson_id="story_generation",
        #     env_config=EnvConfig(
        #         env_class="marin.rl.environments.mock_env.MockEnv",
        #         env_args={"task_type": "story_generation", "seed": 42},
        #     ),
        #     dependencies=[],
        #     sampling_params=story_sampling,
        # ),
        # "gsm8k": LessonConfig(
        #     lesson_id="gsm8k",
        #     env_config=EnvConfig(
        #         env_class="marin.rl.environments.gsm8k_env.GSM8KEnv",
        #         env_args={"seed": 42},
        #     ),
        #     dependencies=[],
        #     # dependencies=[LessonDependency(dependency_id="addition_medium", reward_threshold=0.8)],
        #     sampling_params=default_sampling,
        # ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=1,  # Run full eval after every step
        micro_eval_frequency=9999999,  # Effectively disable micro-eval
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=500,  # for math500
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def rl_train(name: str, experiment_config: ExperimentConfig) -> ExecutorStep:
    hf_config = AutoConfig.from_pretrained(experiment_config.model_config.name)
    config = experiment_config.model_config.config_class.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    model_config = dataclasses.replace(
        config, seq_len=experiment_config.max_output_tokens, tokenizer=experiment_config.model_config.tokenizer
    )

    _ = WandbConfig

    trainer_config = TrainerConfig(
        # wandb is persistently crashing
        tracker=WandbConfig(
            project="marin_post_training",
            name=name,
            tags=["rl", "math", experiment_config.model_config.name.split("/")[-1]],
        ),
        # tracker=TensorboardConfig(
        #     logdir=OutputName("tblogs"),
        # ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        # Set the train batch size to num_rollout_workers * n_generations * n_prompts
        # to ensure we accept an entire training batch from the rollout workers.
        train_batch_size=experiment_config.train_batch_size,
        # microbatch to avoid OOM
        per_device_parallelism=experiment_config.per_device_parallelism,
        num_train_steps=500,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
        # distributed=DistributedConfig(
        #     initialize_jax_distributed=False,
        # )
    )

    opt_config = AdamConfig(
        learning_rate=experiment_config.learning_rate,
        weight_decay=0.0,
        warmup=0,
        lr_schedule="constant",
        max_grad_norm=experiment_config.max_grad_norm,
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=OutputName("rollouts"),
    )
    weight_transfer = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        # We are running on-policy, so wait for new weights from the trainer after each episode.
        max_weight_transfer_wait_time=300,
        coordinator_name=f"weight_transfer_coordinator_{name}",
    )

    curriculum_config = create_math_curriculum(name, experiment_config)

    # Create RLJobConfig using the new unified interface
    config = RLJobConfig(
        model=model_config,
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=opt_config,
            rl_loss=experiment_config.rl_loss,
            replay_buffer=ReplayBufferConfig(
                capacity=4096,
                alpha=3,
                max_samples=1,
                max_rollout_step_delay=experiment_config.max_rollout_step_delay,
            ),
        ),
        curriculum=curriculum_config,
        tokenizer=experiment_config.model_config.tokenizer,
        inference_type="vllm",
        # inference_type="levanter",
        inference_config=vLLMInferenceContextConfig(
            model_name=experiment_config.model_config.name,
            max_model_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
            tensor_parallel_size=8,
            gpu_memory_utilization=0.90,
            sampling_params=SamplingParams(
                temperature=1.0,
                n=8,
                max_tokens=experiment_config.max_output_tokens,
                stop=get_stop_tokens(experiment_config.model_config.type),
                include_stop_str_in_output=True,
                logprobs=1,
            ),
        ),
        initial_checkpoint=experiment_config.model_config.checkpoint,
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        run_id=name,
        log_freq=1,
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            num_train_slices=1,
            num_rollout_workers=1,
            inference_tpu_type="v5p-8",
        ),
        inflight_weight_updates=experiment_config.inflight_weight_updates,
        rollout_tracker=RolloutTrackerConfig(
            project="rl-mockenv-testing",
            name=f"{name}-rollout",
            tags=["rl", "math", "rollout", experiment_config.model_config.name.split("/")[-1]],
        ),
    )

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=RLJob.make_step_fn(),
        config=config,
        pip_dependency_groups=["post_training", "vllm"],
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    llama_8b = ExperimentConfig(
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

    experiment_configs = [llama_8b]
    experiments = []
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for experiment_config in experiment_configs:
        model_base_name = experiment_config.model_config.name.split("/")[-1].lower()
        model_base_name = model_base_name.replace("-instruct", "i")

        # Always include timestamp to avoid cache collisions between runs
        name = f"{model_base_name}-{experiment_config.experiment_name_suffix}-{datestamp}"

        experiments.append(
            rl_train(
                name=name,
                experiment_config=experiment_config,
            ),
        )

    executor_main(
        steps=experiments,
        description="Async RL math training experiments",
    )


if __name__ == "__main__":
    main()
