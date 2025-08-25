"""Configuration dataclasses for Marin RL.

These are separated from :pymod:`marin.rl.datatypes` so that the low-level data
structures remain dependency-free, while configs can import heavier utilities
(e.g. `RayResources`, `ResourceConfig`).
"""

import abc
from dataclasses import dataclass
from typing import Any

from levanter.utils.ray_utils import RayResources
from ray.actor import ActorHandle

from ..resources import ResourceConfig
from .datatypes import InferenceEndpoint, RolloutSink

__all__ = [
    "AbstractEnvConfig",
    "MarinRlConfig",
    "RlTrainingConfig",
]


@dataclass(frozen=True)
class AbstractEnvConfig(abc.ABC):
    """Every environment must expose resource requirements and a build method."""

    @abc.abstractmethod
    def resources(self) -> RayResources:
        """Return Ray resource specs (CPU/GPU/TPU etc.) needed per replica."""

    @abc.abstractmethod
    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, seed: int) -> ActorHandle:
        """Instantiate the environment.

        The *rollout_sink* should be called with :class:`~marin.rl.datatypes.RolloutGroup` batches.
        """


@dataclass(frozen=True)
class RlTrainingConfig:
    """Learner hyper-parameters common to most RL algorithms."""

    num_steps: int
    batch_size: int
    # Extend with learning-rate schedules, clip settings, etc. as needed.


# Placeholders for inference endpoints and environments.
@dataclass(slots=True, frozen=True)
class InferenceConfig:
    """Configuration for inference endpoints.

    This is a placeholder for future extensions (e.g. multiple endpoints,
    failover, etc.).
    """

    endpoint: InferenceEndpoint


@dataclass(frozen=True)
class MarinRlConfig:
    """Root config that ties together env, learner, and infra settings."""

    name: str
    envs: list[AbstractEnvConfig]
    inference: InferenceConfig
    learner: RlTrainingConfig
    learner_resources: ResourceConfig
    # Any user-defined extra fields
    extras: dict[str, Any] | None = None
