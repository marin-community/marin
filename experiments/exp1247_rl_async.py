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
Async RL training experiment for with math environment.
"""

import dataclasses
import datetime
import logging
import os
from dataclasses import dataclass

import jmp
import ray
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.infra.ray_tpu import run_on_pod_ray
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.tensorboard import TensorboardConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from ray.runtime_env import RuntimeEnv
from transformers import AutoConfig, AutoTokenizer

from marin.execution.executor import (
    ExecutorStep,
    OutputName,
    executor_main,
)
from marin.resources import TpuPodConfig
from marin.rl.curriculum import CurriculumConfig, LessonConfig, LessonDependency
from marin.rl.environments import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode
from marin.training.training import (
    _add_run_env_variables,
)
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_TYPE = "llama"
WANDB_PROJECT = f"rl_testing_{MODEL_NAME.split('/')[-1].lower()}"
MODEL_TOKENIZER = MODEL_NAME
MODEL_CHECKPOINT = MODEL_NAME
MAX_INPUT_TOKENS = 128
MAX_OUTPUT_TOKENS = 128
RUN_ID = f"test-{MODEL_NAME.split('/')[-1]}-curriculum"


def stop_tokens(tokenizer_name: str):
    """Infer the stop tokens from the given tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [tokenizer.eos_token_id]


def create_math_curriculum(run_id: str) -> CurriculumConfig:
    """Create progressive math curriculum: comparison -> easy -> medium -> hard."""
    from marin.rl.curriculum import SamplingParams

    # Default sampling params for all lessons
    default_sampling = SamplingParams(
        temperature=0.7,
        n_prompts=16,
        n_generations_per_prompt=16,
        max_tokens=MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS,
        stop_tokens=stop_tokens(MODEL_TOKENIZER),
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
            dependencies=[LessonDependency(dependency_id="number_comparison")],
            sampling_params=default_sampling,
        ),
        "addition_medium": LessonConfig(
            lesson_id="addition_medium",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "addition", "difficulty": "medium", "seed": 42},
            ),
            dependencies=[LessonDependency(dependency_id="addition_easy")],
            sampling_params=default_sampling,
        ),
        "addition_hard": LessonConfig(
            lesson_id="addition_hard",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "addition", "difficulty": "hard", "seed": 42},
            ),
            dependencies=[LessonDependency(dependency_id="addition_medium")],
            sampling_params=default_sampling,
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=100,
        actor_name=f"curriculum-{run_id}",
    )


@dataclass(frozen=True)
class RLTrainConfig:
    """Configuration for async RL training on pods."""

    rollout_worker_config: RolloutWorkerConfig
    train_worker_config: TrainWorkerConfig
    inference_tpu_type: str
    train_tpu_type: str
    num_inference_workers: int
    num_train_slices: int


def run_rl_training_on_pod(config: RLTrainConfig):
    """
    Run async RL training with separate inference and training workers.
    """
    env = {}
    env = _add_run_env_variables(env)
    env["EQX_ON_ERROR"] = "nan"

    # if "JAX_COMPILATION_CACHE_DIR" not in env:
    #     marin_prefix = os.environ.get("MARIN_PREFIX")
    #     if marin_prefix:
    #         env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(marin_prefix, "compilation-cache")
    #         logger.info(f"JAX compilation cache enabled at: {env['JAX_COMPILATION_CACHE_DIR']}")
    #     else:
    #         logger.warning("MARIN_PREFIX environment variable not set. JAX compilation cache will not be configured.")

    runtime_env = RuntimeEnv()

    train_pod_config = TpuPodConfig(tpu_type=config.train_tpu_type, runtime_env=runtime_env)
    rollout_pod_config = TpuPodConfig(tpu_type=config.inference_tpu_type, runtime_env=runtime_env)

    rollout_hw_config = rollout_pod_config.with_env_vars(env)
    train_hw_config = train_pod_config.with_env_vars(env)

    train_kwargs = dict(max_calls=1, **train_hw_config.as_remote_kwargs())
    rollout_kwargs = dict(max_calls=1, **rollout_hw_config.as_remote_kwargs())

    @ray.remote(**train_kwargs)
    def train_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            worker = TrainWorker(
                config=config.train_worker_config,
            )
            worker.train()

    @ray.remote(**rollout_kwargs)
    def inference_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            worker = RolloutWorker(
                config=config.rollout_worker_config,
            )
            worker.run()

    train_tasks = []
    logger.info("Running train worker on TPU type: %s", config.train_tpu_type)
    train_tasks.append(
        run_on_pod_ray.remote(
            train_worker_task,
            config.train_tpu_type,
            num_slices=config.num_train_slices,
            max_retries_failure=10,
            max_retries_preemption=10,
        )
    )

    inference_tasks = []
    for _ in range(config.num_inference_workers):
        logger.info("Running inference worker on TPU type: %s", config.inference_tpu_type)
        inference_tasks.append(
            run_on_pod_ray.remote(
                inference_worker_task,
                config.inference_tpu_type,
                num_slices=1,
                max_retries_failure=10,
                max_retries_preemption=10,
            )
        )

    return ray.get(inference_tasks + train_tasks)


def rl_train(name: str) -> ExecutorStep:
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    config = LlamaConfig.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    model_config = dataclasses.replace(config, seq_len=MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS, tokenizer=MODEL_TOKENIZER)

    _ = WandbConfig

    trainer_config = TrainerConfig(
        # tracker=WandbConfig(
        #     project="marin_rl_testing",
        #     name=name,
        #     tags=["rl", "math", MODEL_NAME.split("/")[-1]],
        # ),
        tracker=TensorboardConfig(
            logdir=OutputName("tblogs"),
        ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=8,
        num_train_steps=50000,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
    )

    opt_config = AdamConfig(
        learning_rate=1e-5,
        weight_decay=1e-2,
        warmup=100,
        lr_schedule="constant",
    )

    inference_server_config = InferenceServerConfig(
        # Turn on tensor parallelism for inference
        trainer=dataclasses.replace(trainer_config, tensor_parallel_axes=["mlp", "kv_head"], model_axis_size=4),
        tokenizer=MODEL_TOKENIZER,
        temperature=1.0,
        service=InferenceEngineConfig(
            max_seqs=16,
            max_pages_per_seq=32,
            page_size=32,
            max_seqs_in_prefill=16,
        ),
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=OutputName("rollouts"),
    )
    weight_transfer = WeightTransferConfig(
        # mode=WeightTransferMode.JAX_TRANSFER_SERVER,
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=4,
        poll_interval_seconds=1,
    )

    curriculum_config = create_math_curriculum(RUN_ID)

    # Create RLJobConfig using the new unified interface
    rl_job_config = RLJobConfig(
        model=model_config,
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=opt_config,
            num_train_steps=50000,
            batch_size=8,
            replay_buffer_capacity=4096,
            replay_buffer_alpha=3,
            max_samples_per_rollout=1,
            max_rollout_delay=32,
        ),
        curriculum=curriculum_config,
        tokenizer=MODEL_TOKENIZER,
        rl_loss=RLOOLoss(kl_coef=0.05, clip_epsilon=0.2),
        initial_checkpoint=MODEL_NAME,
        num_rollout_workers=4,
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        inference_server_config=inference_server_config,
        run_id=RUN_ID,
        log_freq=5,
    )

    # Convert to worker configs for pod deployment
    from marin.rl.rl_job import RLJob

    job = RLJob(rl_job_config)
    train_worker, rollout_worker = job.to_worker_configs()

    config = RLTrainConfig(
        rollout_worker_config=rollout_worker,
        train_worker_config=train_worker,
        inference_tpu_type="v5litepod-4",
        train_tpu_type="v5litepod-4",
        num_inference_workers=4,
        num_train_slices=1,
    )

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=run_rl_training_on_pod,
        config=config,
        pip_dependency_groups=["post_training"],
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    experiments = [
        rl_train(name="llama-1b-math-rl-test-power-012"),
    ]

    executor_main(
        steps=experiments,
        description="Async RL math training experiments",
    )


if __name__ == "__main__":
    main()
