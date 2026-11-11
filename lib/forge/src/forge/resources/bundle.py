# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resources supplied to each pipeline stage."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from shellbox.machine import MachineFactory

from forge.resources.clock import Clock
from forge.resources.decision_model import DecisionModelClient
from forge.resources.env import EnvFactory
from forge.resources.grading import GradingResource
from forge.resources.ids import Ids
from forge.resources.inference import InferenceClient
from forge.resources.rollouts import RolloutResource
from forge.resources.verifier import RoundVerifier
from forge.resources.web import WebClient


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
    role_inference: Mapping[str, InferenceClient] = field(default_factory=dict)

    verifiers: RoundVerifier | None = None
    decision_model: DecisionModelClient | None = None

    async def close_round(self) -> None:
        if self.verifiers is not None:
            await self.verifiers.close()

    def __post_init__(self) -> None:
        object.__setattr__(self, "factories", MappingProxyType(dict(self.factories)))
        object.__setattr__(self, "role_inference", MappingProxyType(dict(self.role_inference)))

    def inference_for(self, role: str) -> InferenceClient:
        return self.role_inference.get(role, self.inference)
