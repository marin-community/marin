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

from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from transformers import AutoConfig
import jmp

from marin.execution.executor import ExecutorStep, executor_main, OutputName
from marin.rl.curriculum import CurriculumConfig, LessonConfig
from marin.rl.environments.base import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import DrGRPOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType

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


llama8b = ModelConfig(
    name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    type="llama",
    tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
    checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct",
    config_class=LlamaConfig,
)

def create_math_curriculum(run_id: str, n_prompts: int, n_generations: int, max_output_length: int, end_of_message_token: int) -> CurriculumConfig:
    """Create progressive math curriculum: comparison -> easy -> medium -> hard."""
    from marin.rl.curriculum import SamplingParams

    # Default sampling params for all lessons
    default_sampling = SamplingParams(
        temperature=1.0,
        max_tokens=max_output_length,
        n_prompts=n_prompts,
        n_generations_per_prompt=n_generations,
        stop_tokens=[end_of_message_token],
    )

    return CurriculumConfig(
        lessons={
            "MATH": LessonConfig(
                lesson_id="MATH",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.tinker_math_env.TinkerMathEnv",
                    env_args={
                        "max_train_examples": None, # Use all
                        "max_eval_examples": 500,
                        "format_coef": 0.1, # Default
                    }
                ),
                dependencies=[],
                sampling_params=default_sampling,
            )
        },
        eval_frequency=1,
        eval_n_examples=500,
        actor_name=f"curriculum-{run_id}",
    )


def rl_train(
    name: str,
    model: ModelConfig,
    end_of_message_token: int,
    tpu_type: str = "v5p-8",
    n_prompts: int = 32,
    n_generations: int = 8,
    kl_coef: float = 0,
    learning_rate: float = 2e-6,
    num_train_steps: int = 300,
    wandb_project: str = "marin_post_training",
    max_input_length: int = 256,
    max_output_length: int = 512,
    **kwargs,
) -> ExecutorStep:
    hf_config = AutoConfig.from_pretrained(model.name)
    config = model.config_class.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    model_config = dataclasses.replace(config, seq_len=max_input_length+max_output_length, tokenizer=model.tokenizer)

    # Generate unique experiment ID with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"{timestamp}"
    run_id = f"{name}-{experiment_id}"

    # Trainer Configuration
    trainer_config = TrainerConfig(
        tracker=WandbConfig(
            project=wandb_project,
            name=run_id,
            id=run_id,
            tags=["rl", "levanter", "math"],
        ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        # Set the train batch size to num_rollout_workers * n_generations * n_prompts
        # to ensure we accept an entire training batch from the rollout workers.
        train_batch_size=n_prompts * n_generations,
        # microbatch to avoid OOM
        per_device_parallelism=16,
        num_train_steps=num_train_steps,
        steps_per_eval=1,
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
        learning_rate=learning_rate,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        max_grad_norm=1.0,
        warmup=0,
        lr_schedule="constant",
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.FILE,
        path=OutputName("rollouts"),
    )

    curriculum_config = create_math_curriculum(run_id, n_prompts, n_generations, max_output_length, end_of_message_token)

    config = RLJobConfig(
        model=model_config,
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=opt_config,
            rl_loss=DrGRPOLoss(kl_coef=kl_coef),
            replay_buffer=ReplayBufferConfig(
                capacity=4096,
                alpha=3,
                max_samples=1,
                max_rollout_step_delay=1,
            ),
        ),
        curriculum=curriculum_config,
        tokenizer=model.tokenizer,
        initial_checkpoint=model.checkpoint,
        rollout_storage=rollout_storage,
        run_id=run_id,
        log_freq=1,
        run_config=RunConfig(
            train_tpu_type=tpu_type,
            num_train_slices=1,
            num_rollout_workers=1,
            inference_tpu_type=tpu_type,
        ),
    )

    # Enable synchronous (on-policy) training mode for testing
    config = config.with_on_policy_training()

    return ExecutorStep(
        name=f"rl_testing/{name}",
        description=f"RL training experiment: {name} for {num_train_steps} steps",
        fn=RLJob.make_step_fn(),
        config=config,
        pip_dependency_groups=["post_training"],
    )


def main():
    """Main function to run RL training experiments."""

    # <|eot_id|>
    end_of_message_token = 128009

    experiments = [
        rl_train(
            name="math500",
            model=llama8b,
            end_of_message_token=end_of_message_token,
            tpu_type="v5p-8",
            n_prompts=8,
            n_generations=4,
            kl_coef=0.0,
            learning_rate=2e-06,
            num_train_steps=300,
            max_output_length=512,
        ),
    ]

    executor_main(
        steps=experiments,
        description="RL math training experiments",
    )


if __name__ == "__main__":
    main()
