"""Common dataclasses and interfaces for Marin's RL subsystem.

This module purposefully contains *only* lightweight type definitions and
interfaces that are shared across the training, environment, and inference
components.  Implementation-specific logic should live elsewhere to avoid
introducing heavy dependencies at import time.
"""

import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from levanter.utils.ray_utils import RayResources

from marin.resources import ResourceConfig

__all__ = [
    "AbstractEnvConfig",
    "MarinRlConfig",
    "RlTrainingConfig",
    "Rollout",
    "RolloutGroup",
    "RolloutSink",
    "Turn",
]


# ---------------------------------------------------------------------------
# Rollouts & Turns
# ---------------------------------------------------------------------------


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
    logprobs: Sequence[float] | None | None
    role: str
    reward: float | None | None
    inference_metadata: dict[str, Any]


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

    rollouts: list[Rollout]
    metadata: dict[str, Any]

    def __iter__(self):
        return iter(self.rollouts)


# A callable that accepts a *batch* of :class:`RolloutGroup` objects.
RolloutSink = Callable[[list[RolloutGroup]], None]


# Placeholders for inference endpoints and environments.
InferenceEndpoint = Any
InferenceConfig = Any
MarinEnv = Any


# ---------------------------------------------------------------------------
# RL Training & Environment configuration stubs
# ---------------------------------------------------------------------------


class AbstractEnvConfig(abc.ABC):
    """Abstract base class that every environment config must implement."""

    @abc.abstractmethod
    def resources(self) -> RayResources:
        """Return the Ray resource requirements to instantiate an environment."""

    @abc.abstractmethod
    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink) -> MarinEnv:
        """Create the environment instance.

        Parameters
        ----------
        inference:
            A handle or address for an OAI-compatible inference endpoint.
        rollout_sink:
            Callable that receives finished rollouts.
        """


@dataclass(slots=True, frozen=True)
class RlTrainingConfig:
    """Minimal learner/training hyperparameters."""

    num_steps: int
    batch_size: int
    # Additional algorithm-specific fields can be added incrementally.


@dataclass(slots=True, frozen=True)
class MarinRlConfig:
    """Top-level configuration object that ties everything together."""

    name: str
    envs: list[AbstractEnvConfig]
    tokenizer: str
    inference: InferenceConfig
    learner: RlTrainingConfig
    learner_resources: ResourceConfig
