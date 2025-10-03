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

import uuid
from dataclasses import dataclass, field

from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from transformers import AutoTokenizer

from marin.rl.curriculum import CurriculumConfig, SamplingParams
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorkerConfig
from marin.rl.train_worker import TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig


@dataclass
class TrainParams:
    """Training configuration parameters."""

    optimizer: OptimizerConfig
    num_train_steps: int

    # Batch sizing
    batch_size: int  # Global batch size (divided across processes)

    # Replay buffer
    replay_buffer_capacity: int = 10000
    replay_buffer_alpha: float = 3.0  # Recency bias
    max_samples_per_rollout: int = 4  # How many times to use each rollout
    max_batch_latency: int = 1000  # Max age of rollouts in steps


def make_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)


@dataclass
class RLJobConfig:
    """Configuration for a complete RL training job."""

    model: LmConfig
    trainer: TrainerConfig
    train_params: TrainParams
    curriculum: CurriculumConfig
    tokenizer: str
    rl_loss: "RLLossModule"

    # Model & initialization (with defaults)
    initial_checkpoint: str | None = None

    # Infrastructure
    num_rollout_workers: int = 1
    rollout_storage: RolloutStorageConfig = field(
        default_factory=lambda: RolloutStorageConfig(
            storage_type=StorageType.FILE,
            queue_name="default",
        )
    )
    weight_transfer: WeightTransferConfig = field(default_factory=WeightTransferConfig)

    # Inference server (auto-configured by default)
    inference_server_config: "InferenceServerConfig | None" = None  # type: ignore

    # Sampling configuration
    eval_sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # Logging
    run_id: str = field(default_factory=lambda: f"rl-{uuid.uuid4().hex[:8]}")
    log_freq: int = 10


class RLJob:
    """High-level interface for RL training jobs.

    Handles worker creation, coordination, and lifecycle management.
    """

    def __init__(self, config: RLJobConfig):
        self.config = config
        self._validate_config()

    def run(self) -> LmHeadModel:
        """Run the RL training job to completion.

        Returns:
            The trained model
        """
        raise NotImplementedError("RLJob.run() will be implemented in Phase 2")

    def to_worker_configs(self) -> tuple[TrainWorkerConfig, RolloutWorkerConfig]:
        """Export worker configurations for inspection/testing.

        Returns:
            Tuple of (TrainWorkerConfig, RolloutWorkerConfig)
        """
        # Create replay buffer config from train params
        replay_buffer = ReplayBufferConfig(
            capacity=self.config.train_params.replay_buffer_capacity,
            alpha=self.config.train_params.replay_buffer_alpha,
            max_samples=self.config.train_params.max_samples_per_rollout,
            max_rollout_delay=self.config.train_params.max_batch_latency,
        )

        # Create inference server config if not provided

        # scan over sampling params for max seqs & tokens etc
        max_seqs = 0
        max_tokens_per_seq = 0
        for lesson in self.config.curriculum.lessons.values():
            total_seqs = lesson.sampling_params.n_generations_per_prompt * lesson.sampling_params.n_prompts
            max_seqs = max(max_seqs, total_seqs)
            max_tokens_per_seq = max(max_tokens_per_seq, lesson.sampling_params.max_tokens)

        if self.config.inference_server_config is None:
            inference_server_config = InferenceServerConfig(
                trainer=self.config.trainer,
                tokenizer=getattr(self.config.model, "tokenizer", None),
                temperature=self.config.eval_sampling_params.temperature,
                service=InferenceEngineConfig(
                    max_seqs=max_seqs,
                    page_size=128,
                    max_pages_per_seq=1 + max_tokens_per_seq // 128,
                    enable_logprobs=True,
                ),
                port=-1,
            )
        else:
            inference_server_config = self.config.inference_server_config

        # Create train worker config
        train_worker_config = TrainWorkerConfig(
            rollout_storage=self.config.rollout_storage,
            weight_transfer=self.config.weight_transfer,
            model=self.config.model,
            trainer=self.config.trainer,
            optimizer=self.config.train_params.optimizer,
            loss=self.config.rl_loss,
            tokenizer=make_tokenizer(self.config.tokenizer),
            replay_buffer=replay_buffer,
            initial_checkpoint=self.config.initial_checkpoint,
            run_id=self.config.run_id,
            curriculum_config=self.config.curriculum,
        )

        # Create rollout worker config
        rollout_worker_config = RolloutWorkerConfig(
            trainer=self.config.trainer,
            inference_server_config=inference_server_config,
            model=self.config.model,
            curriculum_config=self.config.curriculum,
            tokenizer=make_tokenizer(self.config.tokenizer),
            log_freq=self.config.log_freq,
            max_rollouts=None,  # Run indefinitely by default
            initial_checkpoint=self.config.initial_checkpoint,
            weight_transfer=self.config.weight_transfer,
            rollout_storage=self.config.rollout_storage,
            run_id=self.config.run_id,
        )

        return train_worker_config, rollout_worker_config
