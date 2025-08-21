"""Common dataclasses and interfaces for Marin's RL subsystem.

This module purposefully contains *only* lightweight type definitions and
interfaces that are shared across the training, environment, and inference
components.  Implementation-specific logic should live elsewhere to avoid
introducing heavy dependencies at import time.
"""

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from typing import Any

__all__ = [
    "GroupKey",
    "InferenceEndpoint",
    "RLExample",
    "Rollout",
    "RolloutGroup",
    "RolloutRecord",
    "RolloutSink",
    "SampledBatch",
    "Turn",
]


# ---------------------------------------------------------------------------
# Rollouts & Turns
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InferenceMetadata:
    """Metadata about the inference of a turn."""

    model_version: str
    sampling_params: dict[str, Any]
    inference_time: float


@dataclass(slots=True, frozen=True)
class Turn:
    """A single message-level interaction within a rollout.

    Attributes
    ----------
    message:
        The textual contents of the turn (usually a single chat message).
    logprobs:
        Token-level log probabilities corresponding to *message* stored as a
        :class:`numpy.ndarray`. The array length should match the number of
        generated tokens. ``None`` indicates that log probabilities were not
        recorded.
    role:
        The role that produced the message (e.g. "user", "assistant",
        "system", "tool").
    reward:
        Scalar reward obtained *after* the turn.  Can be ``None`` if the reward
        is computed later or not applicable.
    inference_metadata:
        Arbitrary key/value pairs describing how the message was generated
        (model version, sampling parameters, etc.).
    """

    message: str
    logprobs: np.ndarray | None
    role: str
    reward: float | None
    inference_metadata: dict[str, Any] | InferenceMetadata


@dataclass(slots=True, frozen=True)
class Rollout:
    """A sequence of :class:`Turn` objects plus auxiliary metadata."""

    environment: str
    problem_id: str
    rollout_uid: str

    turns: list[Turn]
    metadata: dict[str, Any]

    def __iter__(self):
        return iter(self.turns)


# A callable that accepts a *batch* of :class:`RolloutGroup` objects.
RolloutSink = Callable[[list["RolloutGroup"]], None]


# ---------------------------------------------------------------------------
# Inference endpoint placeholder
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class InferenceEndpoint:
    """Location of an OAI-compatible inference server.

    For now this is just a plain address string (e.g. "http://host:8000" or a
    Ray actor name).  Additional connection metadata can be added later without
    affecting existing code because the dataclass is frozen and explicit.
    """

    address: str


# ---------------------------------------------------------------------------
# Replay buffer data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutRecord:
    """Fully materialized rollout plus associated metadata.

    Attributes
    ----------
    environment:
        Name of the environment that produced the rollout.
    example_id:
        Identifier for the dataset example or task instance.
    policy_version:
        Version of the policy used to generate the rollout.
    replica_id:
        Identifier for the environment replica that produced the rollout.
    rollout_uid:
        Unique identifier for deduplicating rollouts.
    turns:
        Ordered list of :class:`Turn` objects comprising the rollout.
    metadata:
        Additional implementation-defined metadata.
    created_ts:
        UNIX timestamp when the rollout was generated.
    """

    environment: str
    example_id: str
    policy_version: str
    rollout_uid: str
    turns: list[Turn]
    created_ts: float
    metadata: dict[str, Any]
    replica_id: str = "unknown"

    @property
    def total_reward(self):
        return sum(turn.reward for turn in self.turns if turn.reward is not None)


@dataclass(frozen=True)
class RolloutGroup:
    """A sealed collection of rollouts sharing the same group key.

    Attributes
    ----------
    id:
        Deterministic identifier of the group.
    environment:
        Name of the environment that produced the rollouts.
    example_id:
        Identifier of the dataset example shared by all rollouts in the group.
    policy_version:
        Policy version associated with the rollouts.
    rollouts:
        List of :class:`RolloutRecord` objects belonging to the group.
    sealed_ts:
        UNIX timestamp when the group was sealed.
    metadata:
        Additional metadata about the group (e.g. counts, replica info).
    """

    id: str
    environment: str
    example_id: str
    policy_version: str
    rollouts: list[RolloutRecord]
    sealed_ts: float
    metadata: dict[str, Any] = None


@dataclass(frozen=True)
class GroupKey:
    """Key identifying a rollout group before it is sealed."""

    environment: str
    example_id: str


@dataclass(frozen=True)
class SampledBatch:
    """Batch of group identifiers returned by the replay buffer sampler."""

    batch_id: str
    group_ids: list[str]
    ts: float


@dataclass(frozen=True)
class RLExample:
    """A single RL training example.

    Attributes:
        tokens: Token sequence for the example
        loss_mask: Boolean mask indicating which positions to compute loss on
        advantage: Advantage values for each position
        generator_log_probs: Log probabilities from the generator model
    """

    tokens: np.ndarray  # i32["pos"]
    loss_mask: np.ndarray  # bool["pos"]
    advantage: np.ndarray  # float["pos"]
    generator_log_probs: np.ndarray  # float["pos"]
