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
Unified RL job interface for configuring and running RL training.

This module provides a high-level interface that abstracts away worker management
and infrastructure concerns, letting users focus on the RL algorithm and hyperparameters.
"""

import dataclasses
import logging
import os
from typing import Literal
import uuid
from dataclasses import dataclass, field

from marin.rl.environments.inference_ctx import LevanterInferenceContextConfig, vLLMInferenceContextConfig
import ray
from levanter.inference.openai import InferenceServerConfig, InferenceEngineConfig
from levanter.infra.ray_tpu import run_on_pod_ray
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from ray.runtime_env import RuntimeEnv
from transformers import AutoTokenizer, PreTrainedTokenizer

from marin.resources import TpuPodConfig
from marin.rl.curriculum import CurriculumConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig
from marin.training.training import _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for deploying RL workers on TPU pods."""

    train_tpu_type: str
    """TPU type for training workers (e.g., 'v5litepod-4')"""

    num_rollout_workers: int = 4
    """Number of rollout workers to launch"""

    inference_tpu_type: str | None = None
    """TPU type for inference workers. Defaults to train_tpu_type if not specified."""

    num_train_slices: int = 1
    """Number of TPU slices for training worker"""

    max_retries_failure: int = 3
    """Maximum retries on worker failure"""

    max_retries_preemption: int = 100
    """Maximum retries on preemption"""

    runtime_env: RuntimeEnv | None = None
    """Optional Ray runtime environment for workers"""


@dataclass
class TrainParams:
    """RL-specific training configuration parameters."""

    optimizer: OptimizerConfig
    rl_loss: "RLLossModule"
    replay_buffer: ReplayBufferConfig = field(
        default_factory=lambda: ReplayBufferConfig(
            capacity=4096,
            alpha=3.0,
            max_samples=1,
            max_rollout_step_delay=0,
            max_rollout_timestamp_delay=3600.0,
        )
    )


def make_tokenizer(tokenizer: str | PreTrainedTokenizer) -> PreTrainedTokenizer:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


@dataclass
class RLJobConfig:
    """Configuration for a complete RL training job."""

    model: LmConfig
    trainer: TrainerConfig
    train_params: TrainParams
    curriculum: CurriculumConfig
    tokenizer: str | PreTrainedTokenizer

    inference_type: Literal["levanter", "vllm"]

    seed: int = 42

    # Model & initialization (with defaults)
    initial_checkpoint: str | None = None

    # Infrastructure
    rollout_storage: RolloutStorageConfig = field(
        default_factory=lambda: RolloutStorageConfig(
            storage_type=StorageType.FILE,
            queue_name="default",
        )
    )
    weight_transfer: WeightTransferConfig = field(default_factory=WeightTransferConfig)

    # Deployment configuration
    run_config: RunConfig | None = None
    """Configuration for TPU pod deployment. If None, uses simple Ray actors."""

    # Inference server (auto-configured by default)
    inference_config: InferenceServerConfig | vLLMInferenceContextConfig | None = None
    """Configuration for inference context."""

    # Logging
    run_id: str = field(default_factory=lambda: f"rl-{uuid.uuid4().hex[:8]}")
    log_freq: int = 10


class RLJob:
    """High-level interface for RL training jobs.

    Handles worker creation, coordination, and lifecycle management.
    """

    def __init__(self, config: RLJobConfig):
        self.config = config

    # Helper, as Ray doesn't accept method instances for ray.remote
    @staticmethod
    def make_step_fn():
        return lambda config: RLJob(config).run()

    def run(self):
        """Run with TPU pod deployment using run_on_pod_ray."""
        run_config = self.config.run_config
        train_worker_config, rollout_worker_config = self.to_worker_configs()

        # Setup environment
        env = {}
        env = _add_run_env_variables(env)
        env["EQX_ON_ERROR"] = "nan"

        runtime_env = run_config.runtime_env or RuntimeEnv()

        # Create pod configs
        inference_tpu_type = run_config.inference_tpu_type or run_config.train_tpu_type
        train_pod_config = TpuPodConfig(tpu_type=run_config.train_tpu_type, runtime_env=runtime_env)
        rollout_pod_config = TpuPodConfig(tpu_type=inference_tpu_type, runtime_env=runtime_env)

        rollout_hw_config = rollout_pod_config.with_env_vars(env)
        train_hw_config = train_pod_config.with_env_vars(env)

        train_kwargs = dict(max_calls=1, **train_hw_config.as_remote_kwargs())
        rollout_kwargs = dict(max_calls=1, **rollout_hw_config.as_remote_kwargs())

        # Define remote tasks
        @ray.remote(**train_kwargs)
        def train_worker_task():
            with remove_tpu_lockfile_on_exit():
                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                worker = TrainWorker(config=train_worker_config)
                worker.train()

        @ray.remote(**rollout_kwargs)
        def inference_worker_task():
            with remove_tpu_lockfile_on_exit():
                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                # inject a different seed for each worker

                process_id = os.getpid()
                config = dataclasses.replace(
                    rollout_worker_config,
                    seed=rollout_worker_config.seed + process_id,
                    run_id=f"{rollout_worker_config.run_id}-{process_id}",
                )

                worker = RolloutWorker(config=config)
                worker.run()

        train_tasks = [
            run_on_pod_ray.remote(
                train_worker_task,
                run_config.train_tpu_type,
                num_slices=run_config.num_train_slices,
                max_retries_failure=run_config.max_retries_failure,
                max_retries_preemption=run_config.max_retries_preemption,
            )
        ]

        inference_tasks = []
        for _ in range(run_config.num_rollout_workers):
            inference_tasks.append(
                run_on_pod_ray.remote(
                    inference_worker_task,
                    inference_tpu_type,
                    num_slices=1,
                    max_retries_failure=run_config.max_retries_failure,
                    max_retries_preemption=run_config.max_retries_preemption,
                )
            )

        return ray.get(inference_tasks + train_tasks)

    def to_worker_configs(self) -> tuple[TrainWorkerConfig, RolloutWorkerConfig]:
        """Export worker configurations for inspection/testing.

        Returns:
            Tuple of (TrainWorkerConfig, RolloutWorkerConfig)
        """
        # Create tokenizer
        tokenizer = make_tokenizer(self.config.tokenizer)

        # Scan over sampling params for max seqs, must be able to fit a single lesson prompt
        max_seqs = 0
        for lesson in self.config.curriculum.lessons.values():
            total_seqs = lesson.sampling_params.n_generations_per_prompt
            max_seqs = max(max_seqs, total_seqs)

        max_tokens = self.config.curriculum.max_tokens
        assert max_tokens > 0, "Max tokens must be positive across curriculum lessons."

        # Create inference server config if not provided
        if self.config.inference_config is None and self.config.inference_type == "levanter":
            inference_server_config = InferenceServerConfig(
                trainer=dataclasses.replace(
                    self.config.trainer,
                    tensor_parallel_axes=["mlp", "kv_head"],
                ),
                tokenizer=tokenizer,
                temperature=1.0,
                service=InferenceEngineConfig(
                    max_seqs=max_seqs,
                    max_seq_len=max_tokens,
                    page_size=128,
                    hbm_utilization=0.5,
                    enable_logprobs=True,
                ),
                port=0,
            )
            logger.info(
                "Auto-configured InferenceServerConfig for RLJob with max_seqs=%d, max_tokens=%d", max_seqs, max_tokens
            )
            inference_config = LevanterInferenceContextConfig(
                inference_server_config=inference_server_config,
                tokenizer=tokenizer,
                mesh=self.config.trainer.device_mesh,
                axis_mapping=self.config.trainer.compute_axis_mapping,
            )
        else:
            assert self.config.inference_config is not None, "Inference config must be provided for vllm inference"
            inference_config = self.config.inference_config

        # Create train worker config
        train_worker_config = TrainWorkerConfig(
            rollout_storage=self.config.rollout_storage,
            weight_transfer=self.config.weight_transfer,
            model=self.config.model,
            trainer=self.config.trainer,
            optimizer=self.config.train_params.optimizer,
            loss=self.config.train_params.rl_loss,
            tokenizer=tokenizer,
            replay_buffer=self.config.train_params.replay_buffer,
            initial_checkpoint=self.config.initial_checkpoint,
            run_id=self.config.run_id,
            curriculum_config=self.config.curriculum,
            seed=self.config.seed,
        )

        # Create rollout worker config
        rollout_worker_config = RolloutWorkerConfig(
            trainer=self.config.trainer,
            model=self.config.model,
            curriculum_config=self.config.curriculum,
            tokenizer=tokenizer,
            log_freq=self.config.log_freq,
            max_rollouts=None,  # Run indefinitely by default
            initial_checkpoint=self.config.initial_checkpoint,
            weight_transfer=self.config.weight_transfer,
            rollout_storage=self.config.rollout_storage,
            run_id=self.config.run_id,
            seed=self.config.seed + 1000,
            inference_type=self.config.inference_type,
            inference_config=inference_config,
        )

        return train_worker_config, rollout_worker_config
