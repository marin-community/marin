# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

from fray import JobHandle
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import OptimizerConfig
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from levanter.trainer import TrainerConfig
from marin.rl.curriculum import CurriculumConfig
from marin.rl.environments.inference_ctx import (
    LevanterInferenceContextConfig,
    vLLMInferenceContextConfig,
)
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_losses import RLLossModule
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig, RolloutWorkerConfig
from marin.rl.train_worker import TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig
from marin.utilities.json_encoder import CustomJsonEncoder

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

    train_ram: str | None = None
    """Optional host-RAM request override for training workers (e.g. ``"300g"``)."""

    inference_ram: str | None = None
    """Optional host-RAM request override for rollout/inference workers."""

    regions: list[str] | None = None
    """Concrete region(s) to use for all RL worker jobs."""

    zone: str | None = None
    """Concrete zone to use for all RL worker jobs."""

    max_retries_failure: int = 3
    """Maximum retries on worker failure (task code crashes, OOM, etc.)"""

    max_retries_preemption: int = 100
    """Maximum retries on preemption (spot TPU lost, worker node died)"""


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


def make_tokenizer(tokenizer: str | MarinTokenizer) -> MarinTokenizer:
    if isinstance(tokenizer, str):
        return load_tokenizer(tokenizer)
    return tokenizer


@dataclass
class RLJobConfig:
    """Configuration for a complete RL training job."""

    model: LmConfig
    trainer: TrainerConfig
    train_params: TrainParams
    curriculum: CurriculumConfig
    tokenizer: str | MarinTokenizer

    inference_type: Literal["levanter", "vllm"]

    seed: int = 42

    vocab_size: int | None = None
    """Vocab size for model construction. Should match the checkpoint's vocab dimension.
    If None, falls back to tokenizer.vocab_size."""

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
    """Configuration for TPU pod deployment."""

    # Inference server (auto-configured by default)
    inference_config: InferenceServerConfig | vLLMInferenceContextConfig | None = None
    """Configuration for inference context."""

    system_prompt: str | None = None
    """System prompt to use for inference."""

    inflight_weight_updates: bool = False
    """Whether to use inflight weight updates."""

    # Logging
    run_id: str = field(default_factory=lambda: f"rl-{uuid.uuid4().hex[:8]}")
    instance_id: str | None = None
    """Volatile instance identifier, unique per coordinator invocation.

    Used for Iris child job names and hosted actor names so retries after
    preemption do not collide with dead children from previous attempts.
    When ``None``, defaults to :pyattr:`run_id` (legacy single-name behaviour).
    """
    log_freq: int = 10

    rollout_tracker: RolloutTrackerConfig | None = None
    """Tracker configuration for rollout workers. Uses a standalone tracker to avoid JAX deadlocks."""

    pip_dependency_groups: list[str] = field(default_factory=list)
    """Extra pip dependency groups to include for all workers."""

    runtime_env_vars: list[str] = field(default_factory=list)
    """Env var names to copy from the current process into coordinator and worker jobs."""

    @property
    def resolved_instance_id(self) -> str:
        """Return the volatile instance id, falling back to run_id."""
        return self.instance_id if self.instance_id is not None else self.run_id

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

    @staticmethod
    def make_step_fn():
        return lambda config: RLJob(config).run(config.run_id)

    def run(self, name: str) -> JobHandle:
        """Submit the RL job via the v2 orchestration layer.

        Submits a single coordinator job that creates all shared actors
        and child jobs (trainer + rollout workers). The coordinator runs
        inside the cluster with proper job hierarchy.
        """
        from marin.rl.orchestration import submit_rl_job

        handle = submit_rl_job(self.config)
        handle.wait(raise_on_failure=True)
        return handle

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
