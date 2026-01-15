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
import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.distributed import RayConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig
from levanter.utils.mesh import MeshConfig

from marin.execution import step, StepContext, ExecutorStep
from marin.rl.curriculum import CurriculumConfig
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode

from vllm import SamplingParams

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    name: str
    type: str
    tokenizer: str
    checkpoint: str
    config_class: type[HFCompatConfig]
    pip_dependency_groups: list[str] | None = None

    @property
    def safe_name(self) -> str:
        return self.name.replace("/", "-").lower()


@dataclasses.dataclass
class RLExperimentConfig:
    """Shared configuration for RL experiments."""

    model_config: ModelConfig
    rl_loss: RLLossModule
    experiment_name_suffix: str

    # trainer params
    train_batch_size: int = 1024
    per_device_parallelism: int = 16
    num_train_steps: int = 500
    steps_per_eval: int = 100
    checkpointer_save_interval: int = 600

    # wandb
    project_name: str = "marin_post_training"
    tags: list[str] = dataclasses.field(default_factory=lambda: ["rl", "math"])

    # optimization
    learning_rate: float = 1e-7
    max_grad_norm: float = 1.00
    weight_decay: float = 0.0
    warmup: int = 0
    lr_schedule: str = "constant"

    # sampling / tokens
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    n_prompts: int = 64
    n_generations_per_prompt: int = 16

    # replay buffer
    replay_buffer_capacity: int = 4096
    replay_buffer_alpha: float = 3.0
    replay_buffer_max_samples: int = 1

    # execution
    debug_mode: bool = False
    inflight_weight_updates: bool = False
    max_rollout_step_delay: int = 0

    # weight transfer
    weight_transfer_sync_interval_steps: int = 1
    max_weight_transfer_wait_time: int = 300

    # inference context
    inference_tensor_parallel_size: int = 4
    inference_gpu_memory_utilization: float = 0.90
    inference_top_k: int = 4096  # Workaround for vllm-project/tpu-inference#1386
    inference_n: int = 8

    # run config (TPU slice info)
    train_tpu_type: str = "v5p-8"
    inference_tpu_type: str = "v5p-8"
    num_train_slices: int = 1
    num_rollout_workers: int = 1


def get_stop_tokens(model_type: str) -> list[str]:
    """Get model-specific stop tokens."""
    if model_type == "llama":
        return ["<|eot_id|>"]
    elif model_type == "qwen":
        return ["<|im_end|>"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def make_rl_step(name: str, config: RLExperimentConfig, curriculum: CurriculumConfig) -> ExecutorStep:
    @step(
        name=f"rl_testing/{name}",
        description=f"Async RL training: {name}",
        fn=RLJob.make_step_fn(),
        pip_dependency_groups=["vllm", "math"],
    )
    def _step(ctx: StepContext):
        hf_config = AutoConfig.from_pretrained(config.model_config.name)
        model_config_cls = config.model_config.config_class.from_hf_config(hf_config)

        model_config = dataclasses.replace(
            model_config_cls,
            max_seq_len=config.max_input_tokens + config.max_output_tokens,
            tokenizer=config.model_config.tokenizer,
            attn_backend=AttentionBackend.SPLASH,
        )

        tags = [*config.tags, config.model_config.name.split("/")[-1]]

        trainer_config = TrainerConfig(
            tracker=WandbConfig(
                project=config.project_name,
                name=name,
                tags=tags,
            ),
            log_xla_hlo=False,
            log_jaxprs=False,
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=config.train_batch_size,
            per_device_parallelism=config.per_device_parallelism,
            num_train_steps=config.num_train_steps,
            steps_per_eval=config.steps_per_eval,
            checkpointer=CheckpointerConfig(
                base_path=f"{ctx.output}/checkpoints",
                save_interval=datetime.timedelta(seconds=config.checkpointer_save_interval),
            ),
            mesh=MeshConfig(
                axes={"context": 1, "model": 1},
                shared_mapping={"mlp": "model", "heads": "model", "position": "context"},
            ),
            ray=RayConfig(auto_start_cluster=False),
        )

        opt_config = AdamConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup=config.warmup,
            lr_schedule=config.lr_schedule,
            max_grad_norm=config.max_grad_norm,
        )

        rollout_storage = RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"{ctx.output}/rollouts",
        )
        weight_transfer = WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=config.weight_transfer_sync_interval_steps,
            max_weight_transfer_wait_time=config.max_weight_transfer_wait_time,
            coordinator_name=f"weight_transfer_coordinator_{name}",
        )

        # Create RLJobConfig using the unified interface
        job_config = RLJobConfig(
            model=model_config,
            trainer=trainer_config,
            train_params=TrainParams(
                optimizer=opt_config,
                rl_loss=config.rl_loss,
                replay_buffer=ReplayBufferConfig(
                    capacity=config.replay_buffer_capacity,
                    alpha=config.replay_buffer_alpha,
                    max_samples=config.replay_buffer_max_samples,
                    max_rollout_step_delay=config.max_rollout_step_delay,
                ),
            ),
            curriculum=curriculum,
            tokenizer=config.model_config.tokenizer,
            inference_type="vllm",
            inference_config=vLLMInferenceContextConfig(
                model_name=config.model_config.name,
                max_model_len=config.max_input_tokens + config.max_output_tokens,
                tensor_parallel_size=config.inference_tensor_parallel_size,
                gpu_memory_utilization=config.inference_gpu_memory_utilization,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    n=config.inference_n,
                    max_tokens=config.max_output_tokens,
                    stop=get_stop_tokens(config.model_config.type),
                    include_stop_str_in_output=True,
                    logprobs=1,
                    top_k=config.inference_top_k,
                ),
            ),
            initial_checkpoint=config.model_config.checkpoint,
            rollout_storage=rollout_storage,
            weight_transfer=weight_transfer,
            run_id=name,
            log_freq=1,
            run_config=RunConfig(
                train_tpu_type=config.train_tpu_type,
                num_train_slices=config.num_train_slices,
                num_rollout_workers=config.num_rollout_workers,
                inference_tpu_type=config.inference_tpu_type,
            ),
            inflight_weight_updates=config.inflight_weight_updates,
            rollout_tracker=RolloutTrackerConfig(
                project=config.project_name,
                name=f"{name}-rollout",
                tags=[*config.tags, "rollout", config.model_config.name.split("/")[-1]],
            ),
            pip_dependency_groups=(
                config.model_config.pip_dependency_groups
                if config.model_config.pip_dependency_groups
                else ["vllm", "math"]
            ),
        )

        return job_config

    return _step()
