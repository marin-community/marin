# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Type definitions for RL/post-training.

This module contains the shared RL data plane:
- rollout types (Rollout, RolloutGroup, RolloutBatch)
- neutral trajectory and batch types (TrajectoryRecord, SequenceBatch, BatchInfo)
- legacy compatibility types retained for parity checks and helper adapters
  (TrainingBatch, RolloutWithAdvantage)

For inference-related types, see marin.rl.inference_ctx
"""

from dataclasses import dataclass

import equinox as eqx
import haliax.haxtyping as ht
import jax
import numpy as np
from haliax import NamedArray


@dataclass
class RolloutStats:
    """Statistics from a Rollout. Used for curriculum feedback."""

    episode_reward: float
    env_example_id: str
    lesson_id: str
    temperature: float
    top_k: int | None


@dataclass(frozen=True)
class RolloutMetadata:
    """Metadata about when/where rollouts were generated."""

    worker_id: str = ""
    """Worker that generated the rollout."""

    timestamp: float = 0.0
    """The timestamp at which the rollout was generated."""

    weight_step: int = -1
    """The step at which the model weights were used to generate this rollout."""

    run_id: str = ""
    """Stable run identifier for the rollout job."""

    lesson_id: str = ""
    """Lesson identifier associated with this rollout."""

    group_id: str = ""
    """Stable identifier for the rollout group that owns this rollout."""

    trace_id: str = ""
    """Stable identifier for the prompt-level trace that owns this rollout."""

    task_name: str = ""
    """Stable task identifier for this rollout."""

    task_version: str = ""
    """Version for the task contract used to generate this rollout."""

    verifier_name: str | None = None
    """Verifier name associated with this rollout, if any."""

    verifier_version: str | None = None
    """Verifier version associated with this rollout, if any."""

    trace_ref: str | None = None
    """Optional sidecar reference to richer cold-path trace data."""


@dataclass(frozen=True)
class RolloutGroupMetadata:
    """Prompt-level metadata shared across all rollouts in a group."""

    group_id: str = ""
    lesson_id: str = ""
    trace_id: str = ""
    task_name: str = ""
    task_version: str = ""
    verifier_name: str | None = None
    verifier_version: str | None = None
    trace_ref: str | None = None


class Rollout(eqx.Module):
    """A single rollout: one prompt + one generated response + rewards."""

    env_name: str
    """The name of the environment used to generate this rollout."""

    env_example_id: str
    """An identifier for the example used to initialize the environment."""

    prompt_tokens: np.ndarray
    """Array of (prompt_length,) token IDs representing the input prompt."""

    response_tokens: np.ndarray
    """Array of (response_length,) token IDs representing the generated response."""

    response_logprobs: np.ndarray
    """Array of (response_length,) log probabilities for each generated token."""

    token_rewards: np.ndarray
    """The reward assigned to each generated token."""

    episode_reward: float
    """The overall reward for the episode."""

    temperature: float
    """The temperature used to sample the response."""

    top_k: int | None
    """The top_k used to sample the response."""

    is_truncated: bool
    """True if the rollout was truncated due to length. False otherwise."""

    metadata: RolloutMetadata = RolloutMetadata()
    """Metadata about when/where this rollout was generated."""

    correctness_reward: float | None = None
    """The reward for the correctness of the response."""


class RolloutGroup(eqx.Module):
    """Multiple rollouts for the same prompt (e.g., n_generations samples)."""

    rollouts: list[Rollout]
    metadata: RolloutGroupMetadata = RolloutGroupMetadata()


class RolloutBatch(eqx.Module):
    """A batch of rollout groups with metadata."""

    groups: list[RolloutGroup]
    metadata: RolloutMetadata


@dataclass(frozen=True)
class TrajectoryRecord:
    """Objective-neutral per-response training record."""

    trace_id: str
    env_name: str
    task_name: str
    task_version: str
    lesson_id: str
    env_example_id: str
    group_id: str
    verifier_name: str | None
    verifier_version: str | None
    prompt_tokens: np.ndarray
    response_tokens: np.ndarray
    behavior_logprobs: np.ndarray
    token_rewards: np.ndarray
    episode_reward: float
    correctness_reward: float | None
    is_truncated: bool
    sampling_temperature: float
    sampling_top_k: int | None
    rollout_metadata: RolloutMetadata
    trace_ref: str | None = None


@dataclass(frozen=True)
class TrajectoryGroupRecord:
    """Prompt-level grouped trajectory record."""

    group_id: str
    lesson_id: str
    trace_id: str
    task_name: str
    task_version: str
    verifier_name: str | None
    verifier_version: str | None
    prompt_tokens: np.ndarray
    trajectories: tuple[TrajectoryRecord, ...]
    trace_ref: str | None = None


@dataclass(frozen=True)
class BatchInfo:
    """Sidecar metadata carried alongside a SequenceBatch."""

    group_id: tuple[str, ...]
    lesson_id: tuple[str, ...]
    env_name: tuple[str, ...]
    env_example_id: tuple[str, ...]
    task_name: tuple[str, ...]
    task_version: tuple[str, ...]
    verifier_name: tuple[str | None, ...]
    verifier_version: tuple[str | None, ...]
    worker_id: tuple[str, ...]
    run_id: tuple[str, ...]
    trace_id: tuple[str, ...]
    trace_ref: tuple[str | None, ...]
    prompt_length: ht.Int[NamedArray, "batch"]  # noqa: F821
    response_length: ht.Int[NamedArray, "batch"]  # noqa: F821
    episode_reward: ht.Float[NamedArray, "batch"]  # noqa: F821
    token_rewards: ht.Float[NamedArray, "batch position"]
    correctness_reward: ht.Float[NamedArray, "batch"]  # noqa: F821
    weight_step: ht.Int[NamedArray, "batch"]  # noqa: F821
    train_step: ht.Int[NamedArray, "batch"]  # noqa: F821
    timestamp: tuple[float, ...]
    """Wall-clock timestamps in seconds, kept host-side to avoid float32 quantization."""

    def __len__(self) -> int:
        return self.prompt_length.axis_size("batch")


class SequenceBatch(eqx.Module):
    """Objective-neutral batch with only scoring-time tensors."""

    input_ids: ht.Int[NamedArray, "batch position"]
    position_ids: ht.Int[NamedArray, "batch position"]
    prompt_mask: ht.Float[NamedArray, "batch position"]
    response_mask: ht.Float[NamedArray, "batch position"]
    behavior_logprobs: ht.Float[NamedArray, "batch position"]
    sampling_temperature: ht.Float[NamedArray, "batch"]  # noqa: F821
    sampling_top_k: ht.Int[NamedArray, "batch"]  # noqa: F821
    truncated: jax.Array  # [batch]
    max_output_tokens: int

    def __len__(self) -> int:
        return self.input_ids.axis_size("batch")


@dataclass
class RolloutWithAdvantage:
    """Legacy compatibility wrapper retained for parity checks."""

    rollout: Rollout
    advantage: float


class TrainingBatch(eqx.Module):
    """Legacy compatibility batch retained for the old loss implementation."""

    input_ids: ht.Int[NamedArray, "batch position"]
    position_ids: ht.Int[NamedArray, "batch position"]
    loss_weights: ht.Float[NamedArray, "batch position"]
    loss_masks: ht.Int[NamedArray, "batch position"]
    policy_logprobs: ht.Float[NamedArray, "batch position"]
    temperature: ht.Float[NamedArray, "batch"]  # noqa: F821
    top_k: ht.Int[NamedArray, "batch"]  # noqa: F821
    truncated: jax.Array  # [batch] # Make this haxtyped array?
    max_output_tokens: int

    def __len__(self) -> int:
        return self.input_ids.axis_size("batch")
