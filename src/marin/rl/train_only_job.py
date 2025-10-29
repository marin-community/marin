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
Training-only job for benchmarking training throughput in isolation.

This module provides a minimal interface for running pure training workloads
without rollouts, replay buffers, weight transfer, or curriculum logic.
"""

import dataclasses
import logging
import uuid
from dataclasses import dataclass, field

import ray
from levanter.infra.ray_tpu import run_on_pod_ray
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from ray.runtime_env import RuntimeEnv
from transformers import AutoTokenizer, PreTrainedTokenizer

from marin.resources import TpuPodConfig
from marin.rl.rl_losses import RLLossModule
from marin.rl.train_only_worker import TrainOnlyWorker, TrainOnlyWorkerConfig
from marin.training.training import _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


@dataclass
class TrainOnlyRunConfig:
    """Configuration for deploying training-only workers on TPU pods."""

    train_tpu_type: str
    """TPU type for training workers (e.g., 'v5litepod-4')"""

    num_train_slices: int = 1
    """Number of TPU slices for training worker"""

    max_retries_failure: int = 3
    """Maximum retries on worker failure"""

    max_retries_preemption: int = 100
    """Maximum retries on preemption"""

    runtime_env: RuntimeEnv | None = None
    """Optional Ray runtime environment for workers"""


@dataclass
class TrainOnlyParams:
    """Training configuration parameters."""

    optimizer: OptimizerConfig
    rl_loss: RLLossModule


def make_tokenizer(tokenizer: str | PreTrainedTokenizer) -> PreTrainedTokenizer:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


@dataclass
class TrainOnlyJobConfig:
    """Configuration for a training-only benchmark job."""

    model: LmConfig
    trainer: TrainerConfig
    train_params: TrainOnlyParams
    tokenizer: str | PreTrainedTokenizer

    seed: int = 42

    # Model & initialization
    initial_checkpoint: str | None = None

    # Deployment configuration
    run_config: TrainOnlyRunConfig | None = None
    """Configuration for TPU pod deployment. If None, uses simple Ray actors."""

    # Logging
    run_id: str = field(default_factory=lambda: f"train-only-{uuid.uuid4().hex[:8]}")
    log_freq: int = 10

    # Data generation
    sequence_length: int = 1024
    """Fixed sequence length for random token generation."""


class TrainOnlyJob:
    """High-level interface for training-only benchmark jobs.

    This is a simplified version of RLJob that only runs training without
    any RL-specific components like rollouts, replay buffer, or weight transfer.
    """

    def __init__(self, config: TrainOnlyJobConfig):
        self.config = config

    # Helper, as Ray doesn't accept method instances for ray.remote
    @staticmethod
    def make_step_fn():
        return lambda config: TrainOnlyJob(config).run()

    def run(self):
        """Run with TPU pod deployment using run_on_pod_ray."""
        run_config = self.config.run_config
        train_worker_config = self.to_worker_config()

        # Setup environment
        env = {}
        env = _add_run_env_variables(env)
        env["EQX_ON_ERROR"] = "nan"

        runtime_env = run_config.runtime_env or RuntimeEnv()

        # Create pod config
        train_pod_config = TpuPodConfig(tpu_type=run_config.train_tpu_type, runtime_env=runtime_env)
        train_hw_config = train_pod_config.with_env_vars(env)

        train_kwargs = dict(max_calls=1, **train_hw_config.as_remote_kwargs())

        # Define remote task
        @ray.remote(**train_kwargs)
        def train_worker_task():
            with remove_tpu_lockfile_on_exit():
                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                worker = TrainOnlyWorker(config=train_worker_config)
                worker.train()

        train_task = run_on_pod_ray.remote(
            train_worker_task,
            run_config.train_tpu_type,
            num_slices=run_config.num_train_slices,
            max_retries_failure=run_config.max_retries_failure,
            max_retries_preemption=run_config.max_retries_preemption,
        )

        return ray.get([train_task])

    def to_worker_config(self) -> TrainOnlyWorkerConfig:
        """Export worker configuration for inspection/testing.

        Returns:
            TrainOnlyWorkerConfig
        """
        # Create tokenizer
        tokenizer = make_tokenizer(self.config.tokenizer)

        # Create train worker config
        train_worker_config = TrainOnlyWorkerConfig(
            model=self.config.model,
            trainer=self.config.trainer,
            optimizer=self.config.train_params.optimizer,
            loss=self.config.train_params.rl_loss,
            tokenizer=tokenizer,
            initial_checkpoint=self.config.initial_checkpoint,
            run_id=self.config.run_id,
            seed=self.config.seed,
            sequence_length=self.config.sequence_length,
        )

        return train_worker_config
