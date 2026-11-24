# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime composition: Python adapters beside immutable policy data."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from taskcompendium.models import TaskResource

from taskforge.harness.protocol import Harness, Tool
from taskforge.model.review import BandRules, Budget, Decision
from taskforge.model.workpiece import Workpiece
from taskforge.policy import Policy


class ReviewRules(Protocol):
    def __call__(self, workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision: ...


def no_files(_workpiece: Workpiece, _policy: Policy) -> tuple[TaskResource, ...]:
    return ()


@dataclass(frozen=True, slots=True)
class WorkspaceHydration:
    """Files a caller adds to every author workspace, at ``/<path>``; the default adds none.

    ``rendered`` may depend on the workpiece (for example per-proposal reference data) and must be
    deterministic, because a resumed stage replays the write. Only the author sandbox receives these
    files: solvers, adversaries and graders see a file only if the author puts it in the task.
    """

    static: tuple[TaskResource, ...] = ()
    rendered: Callable[[Workpiece, Policy], tuple[TaskResource, ...]] = no_files


@dataclass(frozen=True, slots=True)
class TaskforgeConfig:
    policy: Policy
    author: Harness
    adversaries: Mapping[str, Harness]
    tools: Mapping[str, Mapping[str, Tool[object]]]
    review: ReviewRules
    band: BandRules
    budget: Budget
    workspace: WorkspaceHydration = WorkspaceHydration()

    def __post_init__(self) -> None:
        # A frozen dataclass does not freeze the mappings it holds; read-only views keep the config immutable.
        object.__setattr__(self, "adversaries", MappingProxyType(dict(self.adversaries)))
        object.__setattr__(
            self, "tools", MappingProxyType({name: MappingProxyType(dict(tools)) for name, tools in self.tools.items()})
        )
