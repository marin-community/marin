# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime composition: Python adapters beside immutable policy data."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast

from taskcompendium.models import TaskResource

from taskforge.harness.protocol import Harness, Tool
from taskforge.harness.tools.submit import Gate
from taskforge.model.review import BandRules, Budget, Decision
from taskforge.model.stage import Exit, Stage
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


type Target = Stage | None | Callable[[Workpiece], Stage | None]
type Edge[E: Exit] = tuple[Target, Callable[[Workpiece, E], Workpiece]]
type Transitions = Mapping[tuple[Stage, type[Exit]], Edge[Exit]]


def transition[E: Exit](
    stage: Stage, kind: type[E], target: Target, adapt: Callable[[Workpiece, E], Workpiece]
) -> tuple[tuple[Stage, type[Exit]], Edge[Exit]]:
    """Check adapter types at registration; coordinator dispatches by exact exit class."""
    return (stage, kind), (target, cast(Callable[[Workpiece, Exit], Workpiece], adapt))


@dataclass(frozen=True, slots=True)
class TaskforgeConfig:
    policy: Policy
    author: Harness
    adversaries: Mapping[str, Harness]
    tools: Mapping[str, Mapping[str, Tool[object]]]
    submit_gate: Gate
    review: ReviewRules
    band: BandRules
    budget: Budget
    transitions: Transitions
    workspace: WorkspaceHydration = WorkspaceHydration()

    def __post_init__(self) -> None:
        # A frozen dataclass does not freeze the mappings it holds; read-only views keep the config immutable.
        object.__setattr__(self, "adversaries", MappingProxyType(dict(self.adversaries)))
        object.__setattr__(
            self, "tools", MappingProxyType({name: MappingProxyType(dict(tools)) for name, tools in self.tools.items()})
        )
        object.__setattr__(self, "transitions", MappingProxyType(dict(self.transitions)))
