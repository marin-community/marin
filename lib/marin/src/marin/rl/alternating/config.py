# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for alternating single-pod RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from levanter.utils.fsspec_utils import join_path

DEFAULT_TRAINING_COMPILATION_CACHE_DIR = "/home/levanter/compilation-cache/train"
DEFAULT_SAMPLING_COMPILATION_CACHE_DIR = "/home/levanter/compilation-cache/vllm"

SAMPLER_CONTAINER_NAME = "marin-alt-sampler"
MATERIALIZER_CONTAINER_NAME = "marin-alt-materializer"
TRAINER_CONTAINER_NAME = "marin-alt-trainer"


class AlternatingMode(StrEnum):
    """Top-level entrypoint modes for alternating RL."""

    CONTROLLER = "controller"
    PREPARE_SAMPLING = "prepare-sampling"
    SAMPLING_HOST = "sampling-host"
    MATERIALIZE = "materialize"
    TRAIN_PHASE = "train-phase"
    EXPORT_POLICY = "export-policy"


@dataclass(frozen=True)
class AlternatingClusterConfig:
    """TPU pod layout shared by every phase."""

    tpu_name: str
    tpu_type: str
    zone: str
    num_hosts: int
    local_tensor_parallel_size: int
    node_count: int = 1
    capacity_type: str = "on-demand"
    runtime_version: str | None = None

    def validate(self) -> None:
        """Validate TPU topology inputs."""
        if not self.tpu_name:
            raise ValueError("cluster.tpu_name must be set")
        if not self.tpu_type:
            raise ValueError("cluster.tpu_type must be set")
        if not self.zone:
            raise ValueError("cluster.zone must be set")
        if self.num_hosts < 1:
            raise ValueError(f"cluster.num_hosts must be >= 1, got {self.num_hosts}")
        if self.local_tensor_parallel_size < 1:
            raise ValueError(
                "cluster.local_tensor_parallel_size must be >= 1, " f"got {self.local_tensor_parallel_size}"
            )
        if self.node_count < 1:
            raise ValueError(f"cluster.node_count must be >= 1, got {self.node_count}")


@dataclass(frozen=True)
class AlternatingPhaseQuotaConfig:
    """Sampling and training quotas for one alternating phase."""

    steps_per_phase: int
    num_train_steps: int
    groups_per_training_step: int
    eval_examples_per_lesson: int
    train_groups_per_host: int | None = None
    max_phases: int | None = None

    def validate(self) -> None:
        """Validate quota inputs."""
        if self.steps_per_phase < 1:
            raise ValueError(f"steps_per_phase must be >= 1, got {self.steps_per_phase}")
        if self.num_train_steps < 1:
            raise ValueError(f"num_train_steps must be >= 1, got {self.num_train_steps}")
        if self.groups_per_training_step < 1:
            raise ValueError("groups_per_training_step must be >= 1, " f"got {self.groups_per_training_step}")
        if self.eval_examples_per_lesson < 1:
            raise ValueError("eval_examples_per_lesson must be >= 1, " f"got {self.eval_examples_per_lesson}")
        if self.train_groups_per_host is not None and self.train_groups_per_host < 1:
            raise ValueError("train_groups_per_host must be >= 1 when provided, " f"got {self.train_groups_per_host}")
        if self.max_phases is not None and self.max_phases < 1:
            raise ValueError(f"max_phases must be >= 1 when provided, got {self.max_phases}")

    @property
    def total_train_groups_per_phase(self) -> int:
        """Return the total rollout groups needed for one phase."""
        return self.steps_per_phase * self.groups_per_training_step

    def train_group_targets_by_host(self, *, num_hosts: int) -> list[int]:
        """Return per-host rollout-group targets for the next sampling phase."""
        if num_hosts < 1:
            raise ValueError(f"num_hosts must be >= 1, got {num_hosts}")
        if self.train_groups_per_host is not None:
            return [self.train_groups_per_host] * num_hosts

        total_groups = self.total_train_groups_per_phase
        base_groups = total_groups // num_hosts
        remainder = total_groups % num_hosts
        return [base_groups + int(host_ordinal < remainder) for host_ordinal in range(num_hosts)]


@dataclass(frozen=True)
class AlternatingCacheConfig:
    """Persistent compilation cache paths shared across phase restarts."""

    training_compilation_cache_dir: str = DEFAULT_TRAINING_COMPILATION_CACHE_DIR
    sampling_compilation_cache_dir: str = DEFAULT_SAMPLING_COMPILATION_CACHE_DIR

    def validate(self) -> None:
        """Validate cache paths."""
        if not self.training_compilation_cache_dir:
            raise ValueError("training_compilation_cache_dir must be set")
        if not self.sampling_compilation_cache_dir:
            raise ValueError("sampling_compilation_cache_dir must be set")


@dataclass(frozen=True)
class AlternatingRLConfig:
    """Controller-facing configuration for alternating single-pod RL."""

    run_id: str
    shared_root: str
    image_digest: str
    seed: int
    cluster: AlternatingClusterConfig
    quotas: AlternatingPhaseQuotaConfig
    trainer: Any
    model: Any
    optimizer: Any
    loss: Any
    curriculum: Any
    inference: Any
    replay_buffer: Any
    tokenizer_name: str
    initial_checkpoint: str | None = None
    vocab_size: int | None = None
    system_prompt: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    caches: AlternatingCacheConfig = field(default_factory=AlternatingCacheConfig)

    def validate(self) -> None:
        """Validate the full alternating RL config."""
        if not self.run_id:
            raise ValueError("run_id must be set")
        if not self.shared_root:
            raise ValueError("shared_root must be set")
        if not self.image_digest:
            raise ValueError("image_digest must be set")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must be set")

        self.cluster.validate()
        self.quotas.validate()
        self.caches.validate()

        if getattr(self.inference, "tensor_parallel_size", None) != self.cluster.local_tensor_parallel_size:
            raise ValueError(
                "inference.tensor_parallel_size must match cluster.local_tensor_parallel_size: "
                f"{getattr(self.inference, 'tensor_parallel_size', None)} != "
                f"{self.cluster.local_tensor_parallel_size}"
            )
        if getattr(self.curriculum, "eval_n_examples", None) != self.quotas.eval_examples_per_lesson:
            raise ValueError(
                "curriculum.eval_n_examples must match quotas.eval_examples_per_lesson: "
                f"{getattr(self.curriculum, 'eval_n_examples', None)} != "
                f"{self.quotas.eval_examples_per_lesson}"
            )
        inference_max_model_len = getattr(self.inference, "max_model_len", None)
        if (
            inference_max_model_len is not None
            and getattr(self.curriculum, "max_seq_len", None) != inference_max_model_len
        ):
            raise ValueError(
                "curriculum.max_seq_len must match inference.max_model_len: "
                f"{getattr(self.curriculum, 'max_seq_len', None)} != "
                f"{inference_max_model_len}"
            )
        if self.global_batch_size < 1:
            raise ValueError(f"trainer.train_batch_size must be >= 1, got {self.global_batch_size}")

    @property
    def run_root(self) -> str:
        """Return the per-run root under the shared storage prefix."""
        return join_path(self.shared_root.rstrip("/"), self.run_id)

    @property
    def global_batch_size(self) -> int:
        """Return the global training batch size used during materialization."""
        return int(self.trainer.train_batch_size)
