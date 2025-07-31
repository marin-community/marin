"""Common dataclasses and interfaces for Marin's RL subsystem.

This module purposefully contains *only* lightweight type definitions and
interfaces that are shared across the training, environment, and inference
components.  Implementation-specific logic should live elsewhere to avoid
introducing heavy dependencies at import time.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "InferenceEndpoint",
    "Rollout",
    "RolloutGroup",
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

    Parameters
    ----------
    message:
        The textual contents of the turn (usually a single chat message).
    logprobs:
        Token-level log probabilities corresponding to *message*.  The length of
        *logprobs* should match the number of generated tokens, but can be
        omitted when not available.
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
    logprobs: Sequence[float] | None
    role: str
    reward: float | None | None
    inference_metadata: InferenceMetadata


@dataclass(slots=True, frozen=True)
class Rollout:
    """A sequence of :class:`Turn` objects plus auxiliary metadata."""

    turns: list[Turn]
    metadata: dict[str, Any]

    def __iter__(self):
        return iter(self.turns)


@dataclass(slots=True, frozen=True)
class RolloutGroup:
    """A collection of rollouts that should be processed together (e.g. a batch).

    Grouping rollouts enables algorithms such as GRPO that operate on multiple
    trajectories simultaneously while preserving per-group metadata (problem
    name, seed, etc.).
    """

    id: str
    source: str  # name of the environment that produced the rollouts
    created: float  # POSIX timestamp (seconds since epoch)
    rollouts: list[Rollout]
    metadata: dict[str, Any]

    def __iter__(self):
        return iter(self.rollouts)


# A callable that accepts a *batch* of :class:`RolloutGroup` objects.
RolloutSink = Callable[[list[RolloutGroup]], None]


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


# Placeholders for inference endpoints and environments.
InferenceConfig = Any


# ---------------------------------------------------------------------------
# Configs now reside in :pymod:`marin.rl.config`.
# ---------------------------------------------------------------------------
