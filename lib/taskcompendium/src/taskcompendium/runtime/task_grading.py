# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a captured attempt with whatever grader the task declares."""

import asyncio

from shellbox.machine import MachineFactory, MachineSpec
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import JudgeSpec

from taskcompendium.decision import DecisionClient, DecisionGrader
from taskcompendium.grading import grade_answer, grade_decision_answer, grade_remote_judge_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    GradingAttempt,
    NoGrader,
    SessionGrader,
    TaskSpec,
    VerifyitGrader,
    require_resolved_environment,
    verifyit_spec,
)
from taskcompendium.runtime.environment import validate_machine_spec
from taskcompendium.runtime.grading import grade_in_sandbox


def grade_task(
    task: TaskSpec,
    attempt: GradingAttempt,
    *,
    machine_factory: MachineFactory | None = None,
    machine_spec: MachineSpec | None = None,
    judge_connection: JudgeConnection | None = None,
    decision_client: DecisionClient | None = None,
    decision_timeout: float | None = None,
) -> GradeResult:
    """Grade with an explicit judge connection or decision client, in process, or in a fresh grading machine.

    A ``NoGrader`` task is unavailable. A session grader runs only inside its rollout session. A
    decision-model grader grades on the host with ``decision_client``, its read bounded by
    ``decision_timeout`` seconds; it raises ``ValueError`` without a timeout. Judge, decision-model
    and machine failures, and a missing decision client, become infrastructure errors, except that
    ``DecisionUnavailable`` propagates so the caller can grade once more.
    """
    grader = task.grader
    if isinstance(grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, grader.reason)
    if isinstance(grader, SessionGrader):
        raise TypeError("A session grader runs inside its rollout session")
    if isinstance(grader, DecisionGrader):
        if decision_timeout is None:
            raise ValueError("Decision-model grading requires a decision timeout")
        return asyncio.run(grade_decision_answer(task, attempt, decision_client, timeout=decision_timeout))
    if isinstance(grader, VerifyitGrader) and isinstance(verifyit_spec(grader), JudgeSpec):
        return grade_remote_judge_answer(task, attempt, judge_connection)
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
