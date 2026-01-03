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
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import (
    ExecutorStep,
    OutputName,
    executor_main,
)
from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.environments import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 1024
MAX_SEQ_LEN = 4096 + MAX_OUTPUT_TOKENS


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_type: str
    model_tokenizer: str
    model_checkpoint: str


model_configs = [
    ModelConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_type="llama",
        model_tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        model_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    ),
    ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_type="llama",
        model_tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        model_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
    ),
    ModelConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_type="qwen",
        model_tokenizer="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_checkpoint="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        model_type="qwen",
        model_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        model_checkpoint="Qwen/Qwen3-4B-Instruct-2507",
    ),
]


def stop_tokens(tokenizer_name: str):
    """Infer the stop tokens from the given tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [tokenizer.eos_token_id]


def create_math_curriculum(run_id: str, model_name: str) -> CurriculumConfig:
    from marin.rl.curriculum import SamplingParams

    # Default sampling params for all lessons
    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=8,
        n_generations_per_prompt=8,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        stop_tokens=stop_tokens(model_name),
    )

    lessons = {
        "math": LessonConfig(
            lesson_id="math",
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
        max_seq_len=MAX_SEQ_LEN,
        eval_frequency=100,
        actor_name=f"curriculum-{run_id}",
    )


def rl_train(name: str, model_config: ModelConfig) -> ExecutorStep:
    hf_config = AutoConfig.from_pretrained(model_config.model_name)
    lev_config = LlamaConfig.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    lev_config = dataclasses.replace(lev_config, seq_len=MAX_SEQ_LEN, tokenizer=model_config.model_tokenizer)

    _ = WandbConfig

    trainer_config = TrainerConfig(
        # wandb is persistently crashing
        tracker=WandbConfig(
            project="marin",
            name=name,
            tags=["rl", "math", model_config.model_name.split("/")[-1]],
        ),
        # tracker=TensorboardConfig(
        #     logdir=OutputName("tblogs"),
        # ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        # Set the train batch size to num_rollout_workers * n_generations * n_prompts
        # to ensure we accept an entire training batch from the rollout workers.
        train_batch_size=64,
        # microbatch to avoid OOM
        per_device_parallelism=16,
        num_train_steps=200,
        steps_per_eval=10,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        mesh=MeshConfig(
            axes={"context": 1, "model": 1},  # inherited data:-1, replica:1
            shared_mapping={"mlp": "model", "heads": "model", "position": "context"},
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
    weight_transfer = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        # We are running on-policy, so wait for new weights from the trainer after each episode.
        max_weight_transfer_wait_time=10,
    )

    curriculum_config = create_math_curriculum(name, model_config.model_name)

    lev_config = RLJobConfig(
        inference_type="levanter",
        model=lev_config,
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
        tokenizer=model_config.model_tokenizer,
        initial_checkpoint=model_config.model_checkpoint,
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        run_id=name,
        log_freq=10,
        run_config=RunConfig(
            train_tpu_type="v4-8",
            num_train_slices=1,
            num_rollout_workers=1,
            inference_tpu_type="v4-8",
        ),
    )

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=RLJob.make_step_fn(),
        config=lev_config,
        pip_dependency_groups=["post_training"],
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    experiments = []
    for model_config in model_configs:
        model_base_name = model_config.model_name.split("/")[-1].lower()
        experiments.append(
            rl_train(name=f"{model_base_name}-math-rl-test-chris-{datestamp}", model_config=model_config),
        )

    executor_main(
        steps=experiments,
        description="Async RL math training experiments",
    )


if __name__ == "__main__":
    main()
