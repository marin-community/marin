"""Configuration dataclasses for Marin RL.

These are separated from :pymod:`marin.rl.datatypes` so that the low-level data
structures remain dependency-free, while configs can import heavier utilities
(e.g. `RayResources`, `ResourceConfig`).
"""

import abc
from dataclasses import dataclass
from typing import Any

from levanter.tracker import TrackerConfig
from levanter.utils.ray_utils import RayResources
from ray.actor import ActorHandle

from ..resources import ResourceConfig
from .datatypes import InferenceEndpoint

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
    def build(self, inference: InferenceEndpoint, seed: int) -> ActorHandle:
        """Instantiate the environment actor (pull-based API).

        Returns a Ray actor exposing at least ``step()`` and ``shutdown()``.
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

    pass


@dataclass(frozen=True)
class MarinRlConfig:
    """Root config that ties together env, learner, and infra settings."""
    id: str
    tracker: TrackerConfig
    inference: InferenceConfig

    learner: RlTrainingConfig
    learner_resources: ResourceConfig

    envs: list[AbstractEnvConfig]
    env_replica_counts: dict[str, int] | None = None
    seed: int = 0
