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
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Literal

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from marin.rl.curriculum import CurriculumConfig
from marin.rl.environments.inference_ctx import LevanterInferenceContextConfig, vLLMInferenceContextConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig, RolloutTrackerConfig
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig
from marin.training.training import _add_run_env_variables
from marin.utilities.json_encoder import CustomJsonEncoder
from marin.utils import remove_tpu_lockfile_on_exit
from transformers import AutoTokenizer, PreTrainedTokenizer

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

    env_vars: dict[str, str] = field(default_factory=dict)
    """Custom environment variables for workers"""


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

    vocab_size: int | None = None
    """Vocab size for model construction. Should match the checkpoint's vocab dimension.
    If None, falls back to len(tokenizer)."""

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

    system_prompt: str | None = None
    """System prompt to use for inference."""

    inflight_weight_updates: bool = False
    """Whether to use inflight weight updates."""

    # Logging
    run_id: str = field(default_factory=lambda: f"rl-{uuid.uuid4().hex[:8]}")
    log_freq: int = 10

    rollout_tracker: RolloutTrackerConfig | None = None
    """Tracker configuration for rollout workers. Uses a standalone tracker to avoid JAX deadlocks."""

    pip_dependency_groups: list[str] = field(default_factory=list)
    """Extra pip dependency groups to include for all workers."""

    def with_on_policy_training(self) -> "RLJobConfig":
        """Configure for on-policy training.

        Returns a new RLJob configured to run the inference and training workers
        in lockstep for on-policy training.
        Returns:
            New RLJobConfig configured for synchronous training mode.
        """
        # Update replay buffer to only accept fresh rollouts
        updated_replay_buffer = dataclasses.replace(
            self.train_params.replay_buffer,
            max_rollout_step_delay=0,
            max_samples=1,
        )
        updated_train_params = dataclasses.replace(
            self.train_params,
            replay_buffer=updated_replay_buffer,
        )

        # Update weight transfer to sync every step and wait for new weights
        updated_weight_transfer = dataclasses.replace(
            self.weight_transfer,
            sync_interval_steps=1,
            max_weight_transfer_wait_time=600,
        )

        return dataclasses.replace(
            self,
            train_params=updated_train_params,
            weight_transfer=updated_weight_transfer,
        )


class RLJob:
    """High-level interface for RL training jobs.

    Handles worker creation, coordination, and lifecycle management.
    """

    def __init__(self, config: RLJobConfig):
        self.config = config

    # Helper, as Ray doesn't accept method instances
    @staticmethod
    def make_step_fn():
        return lambda config: RLJob(config).run(config.run_id)

    def run(self, name: str):
        """Run with TPU pod deployment."""
        run_config = self.config.run_config
        train_worker_config, rollout_worker_config = self.to_worker_configs()

        # Setup environment
        env = {"EQX_ON_ERROR": "nan"}
        env = _add_run_env_variables(env)

        # Create resource configs
        inference_tpu_type = run_config.inference_tpu_type or run_config.train_tpu_type
        train_resources = ResourceConfig.with_tpu(run_config.train_tpu_type)
        rollout_resources = ResourceConfig.with_tpu(inference_tpu_type)

        def train_worker_task():
            with remove_tpu_lockfile_on_exit():
                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                worker = TrainWorker(config=train_worker_config)
                worker.train()

        def make_inference_task(worker_idx: int):
            def inference_worker_task():
                with remove_tpu_lockfile_on_exit():
                    logging.basicConfig(
                        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
                    )
                    # use deterministic seed based on worker index

                    config = dataclasses.replace(
                        rollout_worker_config,
                        seed=rollout_worker_config.seed + worker_idx,
                        run_id=f"{rollout_worker_config.run_id}-rollout-{worker_idx}",
                        worker_index=worker_idx,
                    )

                    worker = RolloutWorker(config=config)
                    worker.run()

            return inference_worker_task

        cluster = current_cluster()
        jobs = []
        jobs.append(
            cluster.launch(
                JobRequest(
                    name=f"rl-train-{name}-train",
                    resources=train_resources,
                    entrypoint=Entrypoint.from_callable(train_worker_task),
                    environment=EnvironmentConfig.create(env_vars=env, extras=self.config.pip_dependency_groups),
                )
            )
        )

        for i in range(run_config.num_rollout_workers):
            jobs.append(
                cluster.launch(
                    JobRequest(
                        name=f"rl-train-{name}-rollout-{i}",
                        resources=rollout_resources,
                        entrypoint=Entrypoint.from_callable(make_inference_task(i)),
                        environment=EnvironmentConfig.create(env_vars=env, extras=self.config.pip_dependency_groups),
                    )
                )
            )

        return cluster.wait(jobs, raise_on_failure=True)

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

        max_seq_len = self.config.curriculum.max_seq_len
        assert max_seq_len > 0, "Max seq len must be positive across curriculum lessons."

        # create a unique name for the weight-transfer coordinator based on our config hash
        # this ensures we get the same name across multiple calls
        config_json = json.dumps(dataclasses.asdict(self.config.weight_transfer), sort_keys=True, cls=CustomJsonEncoder)

        config_hash = hashlib.md5(config_json.encode("utf-8")).hexdigest()[:8]

        weight_transfer_coordinator_name = f"wt-coord-{config_hash}"
        weight_transfer_config = dataclasses.replace(
            self.config.weight_transfer,
            coordinator_name=weight_transfer_coordinator_name,
        )

        # Create inference server config if not provided
        if self.config.inference_config is None and self.config.inference_type == "levanter":
            inference_server_config = InferenceServerConfig(
                trainer=self.config.trainer,
                tokenizer=tokenizer,
                temperature=1.0,
                service=InferenceEngineConfig(
                    max_seqs=max_seqs,
                    max_seq_len=max_seq_len,
                    page_size=128,
                    hbm_utilization=0.5,
                ),
                port=0,
            )
            logger.info(
                "Auto-configured InferenceServerConfig for RLJob with max_seqs=%d, max_seq_len=%d", max_seqs, max_seq_len
            )
            inference_config = LevanterInferenceContextConfig(
                mesh=self.config.trainer.device_mesh,
                inference_server_config=inference_server_config,
                tokenizer=tokenizer,
                axis_mapping=self.config.trainer.compute_axis_mapping,
            )
        else:
            assert self.config.inference_config is not None, "Inference config must be provided for vllm inference"
            inference_config = self.config.inference_config

        # Create train worker config
        train_worker_config = TrainWorkerConfig(
            rollout_storage=self.config.rollout_storage,
            weight_transfer=weight_transfer_config,
            model=self.config.model,
            trainer=self.config.trainer,
            optimizer=self.config.train_params.optimizer,
            loss=self.config.train_params.rl_loss,
            tokenizer=tokenizer,
            replay_buffer=self.config.train_params.replay_buffer,
            initial_checkpoint=self.config.initial_checkpoint,
            vocab_size=self.config.vocab_size,
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
            vocab_size=self.config.vocab_size,
            weight_transfer=weight_transfer_config,
            rollout_storage=self.config.rollout_storage,
            run_id=self.config.run_id,
            seed=self.config.seed + 1000,
            inference_type=self.config.inference_type,
            inference_config=inference_config,
            system_prompt=self.config.system_prompt,
            inflight_weight_updates=self.config.inflight_weight_updates,
            tracker_config=self.config.rollout_tracker,
        )

        return train_worker_config, rollout_worker_config
