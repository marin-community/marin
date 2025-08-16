"""Common dataclasses and interfaces for Marin's RL subsystem.

This module purposefully contains *only* lightweight type definitions and
interfaces that are shared across the training, environment, and inference
components.  Implementation-specific logic should live elsewhere to avoid
introducing heavy dependencies at import time.
"""

from collections.abc import Callable
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from typing import Any, Optional

__all__ = [
    "InferenceEndpoint",
    "Rollout",
    "RolloutRecord",
    "RolloutGroup",
    "GroupKey",
    "SampledBatch",
    "RolloutSink",
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
    segment_idx:
        Index of the segment within a multi-part rollout (default ``0``).
    is_last_segment:
        ``True`` if this is the final segment of the rollout.
    replica_id:
        Identifier for the environment replica that produced the rollout.
    rollout_uid:
        Unique identifier for deduplicating rollouts.
    reward:
        Scalar reward for the rollout (if available).
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
    turns: list[Turn]
    created_ts: float
    segment_idx: int = 0
    is_last_segment: bool = True
    replica_id: str = "unknown"
    rollout_uid: str = ""
    reward: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None


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
    segment_idx:
        Segment index for multi-part rollouts.
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
    segment_idx: int
    rollouts: list[RolloutRecord]
    sealed_ts: float
    metadata: dict


@dataclass(frozen=True)
class GroupKey:
    """Key identifying a rollout group before it is sealed."""

    environment: str
    example_id: str
    policy_version: str
    segment_idx: int


@dataclass(frozen=True)
class SampledBatch:
    """Batch of group identifiers returned by the replay buffer sampler."""

    batch_id: str
    group_ids: list[str]
    ts: float
