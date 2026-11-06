# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resources supplied to each pipeline stage."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shellbox.machine import MachineFactory

from forge.harness.web import WebClient
from forge.resources.clock import Clock
from forge.resources.env import EnvFactory
from forge.resources.grading import GradingResource
from forge.resources.ids import Ids
from forge.resources.inference import InferenceClient
from forge.resources.rollouts import RolloutResource


@dataclass(frozen=True, slots=True)
class Resources:
    envs: EnvFactory
    inference: InferenceClient
    rollouts: RolloutResource
    grading: GradingResource
    clock: Clock
    ids: Ids
    factories: Mapping[str, MachineFactory]
    web: WebClient

    def __post_init__(self) -> None:
        object.__setattr__(self, "factories", MappingProxyType(dict(self.factories)))
