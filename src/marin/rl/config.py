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

"""Configuration dataclasses for Marin RL.

These are separated from :pymod:`marin.rl.types` so that the low-level data
structures remain dependency-free, while configs can import heavier utilities
(e.g. `RayResources`, `ResourceConfig`).
"""

import abc
from dataclasses import dataclass
from typing import Any

from levanter.utils.ray_utils import RayResources
from ray.actor import ActorHandle

from ..resources import ResourceConfig
from .types import InferenceConfig, InferenceEndpoint, RolloutSink

__all__ = [
    "AbstractEnvConfig",
    "MarinRlConfig",
    "RlTrainingConfig",
]


class AbstractEnvConfig(abc.ABC):
    """Every environment must expose resource requirements and a build method."""

    @abc.abstractmethod
    def resources(self) -> RayResources:
        """Return Ray resource specs (CPU/GPU/TPU etc.) needed per replica."""

    @abc.abstractmethod
    def build(self, inference: InferenceEndpoint, rollout_sink: RolloutSink, seed: int) -> ActorHandle:
        """Instantiate the environment.

        The *rollout_sink* should be called with :class:`~marin.rl.types.RolloutGroup` batches.
        """


@dataclass(slots=True, frozen=True)
class RlTrainingConfig:
    """Learner hyper-parameters common to most RL algorithms."""

    num_steps: int
    batch_size: int
    # Extend with learning-rate schedules, clip settings, etc. as needed.


@dataclass(slots=True, frozen=True)
class MarinRlConfig:
    """Root config that ties together env, learner, and infra settings."""

    name: str
    envs: list[AbstractEnvConfig]
    inference: InferenceConfig
    learner: RlTrainingConfig
    learner_resources: ResourceConfig
    # Any user-defined extra fields
    extras: dict[str, Any] | None = None
