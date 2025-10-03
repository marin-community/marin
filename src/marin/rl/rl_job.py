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

import ray
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

from marin.rl.curriculum import CurriculumConfig, SamplingParams
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
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


def make_tokenizer(tokenizer: str | PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Create or return tokenizer instance.

    Args:
        tokenizer: Either a HuggingFace model name string or a PreTrainedTokenizer instance

    Returns:
        PreTrainedTokenizer instance
    """
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
        # Get worker configurations
        train_config, rollout_config = self.to_worker_configs()

        # Create Ray remote tasks for workers
        @ray.remote
        def train_worker_task():
            worker = TrainWorker(config=train_config)
            worker.train()

        @ray.remote
        def rollout_worker_task():
            worker = RolloutWorker(config=rollout_config)
            worker.run()

        # Launch training worker
        train_task = train_worker_task.remote()

        # Launch rollout workers
        rollout_tasks = []
        for _ in range(self.config.num_rollout_workers):
            rollout_tasks.append(rollout_worker_task.remote())

        # Wait for training to complete
        try:
            ray.get(train_task)
        finally:
            # Training completed or failed, stop rollout workers
            # Note: rollout workers will stop naturally when training completes
            # We don't need to explicitly stop them as they'll hit max rollouts or timeout
            pass

        # TODO: Return the trained model
        # For now, training completion is signaled by train_task finishing
        raise NotImplementedError("Model retrieval from training worker not yet implemented")

    def to_worker_configs(self) -> tuple[TrainWorkerConfig, RolloutWorkerConfig]:
        """Export worker configurations for inspection/testing.

        Returns:
            Tuple of (TrainWorkerConfig, RolloutWorkerConfig)
        """
        # Create tokenizer
        tokenizer = make_tokenizer(self.config.tokenizer)

        # Create replay buffer config from train params
        replay_buffer = ReplayBufferConfig(
            capacity=self.config.train_params.replay_buffer_capacity,
            alpha=self.config.train_params.replay_buffer_alpha,
            max_samples=self.config.train_params.max_samples_per_rollout,
            max_rollout_delay=self.config.train_params.max_batch_latency,
        )

        # Scan over sampling params for max seqs & tokens
        max_seqs = 0
        max_tokens_per_seq = 0
        for lesson in self.config.curriculum.lessons.values():
            total_seqs = lesson.sampling_params.n_generations_per_prompt * lesson.sampling_params.n_prompts
            max_seqs = max(max_seqs, total_seqs)
            max_tokens_per_seq = max(max_tokens_per_seq, lesson.sampling_params.max_tokens)

        # Create inference server config if not provided
        if self.config.inference_server_config is None:
            inference_server_config = InferenceServerConfig(
                trainer=self.config.trainer,
                tokenizer=tokenizer,
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
            tokenizer=tokenizer,
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
            tokenizer=tokenizer,
            log_freq=self.config.log_freq,
            max_rollouts=None,  # Run indefinitely by default
            initial_checkpoint=self.config.initial_checkpoint,
            weight_transfer=self.config.weight_transfer,
            rollout_storage=self.config.rollout_storage,
            run_id=self.config.run_id,
        )

        return train_worker_config, rollout_worker_config

    def _validate_config(self):
        """Validate configuration consistency."""
        # TODO: Add validation logic
        # - Check curriculum dependencies form DAG
        # - Validate batch_size >= num processes
        # - Ensure all lessons have sampling_params
        pass
