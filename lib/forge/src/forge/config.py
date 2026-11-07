# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime composition: Python adapters beside immutable policy data."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from taskcompendium.models import TaskResource

from forge.harness.protocol import Harness, Tool
from forge.model.review import BandRules, Budget, Decision, ValidationResult
from forge.model.stage import Exit, Stage
from forge.model.workpiece import Submission, Workpiece
from forge.policy import Policy


class SubmitGate(Protocol):
    def __call__(self, submission: Submission) -> tuple[str, ...]: ...


class ReviewRules(Protocol):
    def __call__(self, result: ValidationResult, band: BandRules) -> Decision: ...


@dataclass(frozen=True, slots=True)
class WorkspaceHydration:
    static: tuple[TaskResource, ...]
    rendered: Callable[[Workpiece, Policy], tuple[TaskResource, ...]]


type Edge = tuple[Stage, Callable[[Workpiece, Exit], Workpiece]]
type Transitions = Mapping[tuple[Stage, type[Exit]], Edge]


@dataclass(frozen=True, slots=True)
class ForgeConfig:
    policy: Policy
    author: Harness
    adversaries: Mapping[str, Harness]
    workspace: WorkspaceHydration
    tools: Mapping[str, Mapping[str, Tool[object]]]
    submit_gate: SubmitGate
    review: ReviewRules
    band: BandRules
    budget: Budget
    transitions: Transitions

    def __post_init__(self) -> None:
        object.__setattr__(self, "adversaries", MappingProxyType(dict(self.adversaries)))
        object.__setattr__(
            self, "tools", MappingProxyType({name: MappingProxyType(dict(tools)) for name, tools in self.tools.items()})
        )
        object.__setattr__(self, "transitions", MappingProxyType(dict(self.transitions)))
