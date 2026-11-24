# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime composition: Python adapters beside immutable policy data."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from taskcompendium.models import TaskResource

from taskforge.harness.role import Role
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

    ``rendered`` may depend on the workpiece (for example per-proposal reference data); it has no
    identity, so a run that sets it resumes no stage. Only the author sandbox receives these
    files: solvers, adversaries and graders see a file only if the author puts it in the task.
    """

    static: tuple[TaskResource, ...] = ()
    rendered: Callable[[Workpiece, Policy], tuple[TaskResource, ...]] = no_files


@dataclass(frozen=True, slots=True)
class TaskforgeConfig:
    policy: Policy
    author: Role
    adversary: Role
    """Every name on ``adversary.roster`` runs this role."""
    solve_review: Role
    """The machine-less reviewer of every reviewable validation round; its tools are bound per visit."""
    review: ReviewRules
    band: BandRules
    budget: Budget
    workspace: WorkspaceHydration = WorkspaceHydration()
