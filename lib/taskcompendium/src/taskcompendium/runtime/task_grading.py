# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a captured attempt with whatever grader the task declares."""

import asyncio

from shellbox.machine import MachineFactory, MachineSpec

from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    GradingAttempt,
    NoGrader,
    SessionGrader,
    TaskSpec,
    require_resolved_environment,
)
from taskcompendium.runtime.environment import validate_machine_spec
from taskcompendium.runtime.grading import grade_in_sandbox


def grade_task(
    task: TaskSpec,
    attempt: GradingAttempt,
    *,
    machine_factory: MachineFactory | None = None,
    machine_spec: MachineSpec | None = None,
) -> GradeResult:
    """Grade in process when the grader allows it, otherwise in a fresh machine from ``machine_factory``.

    A ``NoGrader`` task is unavailable. A session grader runs only inside its rollout session.
    Machine failures become infrastructure errors.
    """
    grader = task.grader
    if isinstance(grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, grader.reason)
    if isinstance(grader, SessionGrader):
        raise TypeError("A session grader runs inside its rollout session")
    if grader.environment is None:
        return grade_answer(task, attempt)
    require_resolved_environment(grader.environment)
    if machine_factory is None or machine_spec is None:
        return GradeResult(Outcome.INFRA_ERROR, None, "Sandbox grading requires a machine factory and specification")
    validate_machine_spec(grader.environment, machine_factory, machine_spec)
    return asyncio.run(sandbox_grade(task, attempt, machine_factory, machine_spec))


async def sandbox_grade(
    task: TaskSpec,
    attempt: GradingAttempt,
    machine_factory: MachineFactory,
    machine_spec: MachineSpec,
    *,
    timeout: float | None = None,
) -> GradeResult:
    """Run ``grade_in_sandbox``, reporting machine failures as infrastructure errors instead of raising."""
    try:
        return await grade_in_sandbox(task, attempt, machine_factory, machine_spec, timeout=timeout)
    except (RuntimeError, OSError) as error:
        return GradeResult(Outcome.INFRA_ERROR, None, str(error))
