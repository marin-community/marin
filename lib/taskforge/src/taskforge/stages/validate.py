# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""


import asyncio
from pathlib import PurePosixPath

from rolloutengine.models import TaskRolloutSpec, TraceMode
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note, Resolved
from taskforge.effects.issues import Hold, OutputBudgetExhausted, describe_exception
from taskforge.harness.protocol import Returned, Stopped, StopReason
from taskforge.harness.role import Fresh, Visit, sitting
from taskforge.harness.tools.adversary import AdversaryClaim, AdversaryReport, ReportTool
from taskforge.lowering import (
    InvalidTaskSpec,
    lower_with_settings,
    lowering_settings,
    requirements,
)
from taskforge.model.controls import Control
from taskforge.model.proposal import render
from taskforge.model.review import (
    AdversaryFinding,
    AdversaryResult,
    AdversaryStatus,
    Cause,
    ControlResult,
    Modify,
    Observation,
    ObservedStatus,
    SearchLimit,
    TrialResult,
    ValidationResult,
)
from taskforge.model.workpiece import Workpiece
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import sandbox_limits
from taskforge.stages.classify import CAUSE_DECISIONS, grade_cause, grader_infra_unavailable, trial_cause
from taskforge.stages.exits import Validated
from taskforge.stages.summary import task_manifest


async def control_result(ctx: Context, spec: TaskRolloutSpec, index: int, control: Control) -> ControlResult:
    grade = await ctx.emit(GradeCandidate(spec, control.candidate))
    cause = grade_cause(grade, spec.task)
    return ControlResult(index, grade, cause is None and control.expected.met_by(grade), cause)


async def solver_result(ctx: Context, spec: TaskRolloutSpec, trial: int, attempt: int) -> TrialResult:
    for retry in range(2):
        result = None
        try:
            result = await ctx.emit(SolverRollout(spec, trial, attempt))
        except* OutputBudgetExhausted:
            ctx.notify(Note(f"Solver {trial}: output budget exhausted on attempt {retry + 1} of 2"))
        if result is not None:
            return TrialResult(trial, result.grade, trial_cause(result, spec.task))
    return TrialResult(trial, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted"))


async def adversary_result(
    ctx: Context,
    workpiece: Workpiece,
    config: TaskforgeConfig,
    resources: Resources,
    spec: TaskRolloutSpec,
    name: str,
    occurrence: int,
) -> AdversaryResult:
    """One roster entry, ``occurrence`` among entries with its name; each search is its own environment owner."""
    visit = f"validate/{workpiece.round}/{len(workpiece.evidence)}"
    for retry in range(2):
        result = None
        try:
            owner = f"{visit}/adversary:{name}/{occurrence}/{retry}"
            result = await run_adversary(ctx, workpiece, config, resources, spec, name, owner)
        except* Hold as error:
            result = AdversaryResult(name, AdversaryStatus.FAILED, (), failure_reasons(error))
        if result is not None:
            return result
        ctx.notify(Note(f"Adversary {name}: output budget exhausted on attempt {retry + 1} of 2"))
    return exhausted(name, SearchLimit.OUTPUT_BUDGET)


def exhausted(name: str, limit: SearchLimit) -> AdversaryResult:
    reason = (
        f"ran out of its {limit} without reporting a defect; its search budget ran out, it did not validate the task"
    )
    return AdversaryResult(name, AdversaryStatus.EXHAUSTED, (), (reason,), limit)


def failure_reasons(error: BaseException) -> tuple[str, ...]:
    """Retain leaf diagnostics when observer scopes group infrastructure failures."""
    if isinstance(error, BaseExceptionGroup):
        return tuple(reason for child in error.exceptions for reason in failure_reasons(child))
    reasons = (describe_exception(error),)
    return reasons + (failure_reasons(error.__cause__) if error.__cause__ is not None else ())


def observation(grade: GradeResult, task: TaskSpec) -> Observation | None:
    """What a replay shows about the grader; an infrastructure failure shows nothing."""
    cause = grade_cause(grade, task)
    if cause is not None:
        return Observation(ObservedStatus.GRADER_ERROR) if CAUSE_DECISIONS.get(cause) is Modify else None
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return Observation(ObservedStatus.SUBMISSION_FAILURE)
    return Observation(ObservedStatus.GRADED, grade.reward)


def shows(claim: AdversaryClaim, grade: GradeResult, task: TaskSpec) -> bool:
    return (observed := observation(grade, task)) is not None and claim.defect.shown_by(observed)


async def claim_result(ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim) -> AdversaryFinding:
    """The one reproduction check of every brief: the defect shows on the round's verifier, then on a fresh one."""
    grade = await ctx.emit(GradeCandidate(spec, claim.candidate))
    if shows(claim, grade, spec.task):
        grade = await ctx.emit(GradeCandidate(spec, claim.candidate, fresh_verifier=True))
    met = shows(claim, grade, spec.task)  # on the fresh verifier's grade: the first one did not show it otherwise
    return AdversaryFinding(claim.claim, claim.candidate, claim.defect, grade, met, grade_cause(grade, spec.task))


async def reproduced(
    ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim, index: int
) -> tuple[AdversaryFinding, tuple[str, ...]]:
    """Reproduce one claim; an inconclusive reproduction keeps the claim and candidate ungraded, with why.

    Its own scope collects observer holds on its grades, so a sibling claim's grade survives them.
    """
    finding, reasons = AdversaryFinding(claim.claim, claim.candidate, claim.defect, None, False), ()
    try:
        async with ctx.scope(f"claim:{index}"):
            graded = await claim_result(ctx, spec, claim)
        finding = graded
    except* (OutputBudgetExhausted, Hold) as error:
        reasons = failure_reasons(error)
    return finding, reasons


async def run_adversary(
    ctx: Context,
    workpiece: Workpiece,
    config: TaskforgeConfig,
    resources: Resources,
    spec: TaskRolloutSpec,
    name: str,
    owner: str,
) -> AdversaryResult | None:
    """One adversary search and the reproduction of its claims; ``None`` when the search ran out of output."""
    workdir = config.policy.setting("author.workdir", str)
    machine = (
        spec.task.environment_requirements
        if spec.task_machine is not None
        else requirements(image=config.policy.setting("adversary.image", str) or None, workdir=workdir)
    )
    shared = (*spec.task.resources.all, *spec.task.resources.worker)
    files = (
        *((PurePosixPath("/", file.path), resource_bytes(file)) for file in shared),
        (PurePosixPath(workdir, "task.json"), spec.task.model_dump_json().encode()),
    )
    opening = Fresh(
        config.policy.prompt("adversary.system").render(sandbox_limits=sandbox_limits(config.policy)),
        config.policy.prompt("adversary.user").render(
            task=task_manifest(spec.task),
            proposal=render(workpiece.proposal),
            brief=config.policy.prompt(f"adversary.brief.{name}").template,
            workdir=workdir,
        ),
    )
    visit = Visit(f"adversary:{name}", owner, machine, opening, files, {"report": ReportTool(spec.task)})
    async with sitting(ctx, config.adversary, visit, resources) as session:
        try:
            outcome = await session()
        except* OutputBudgetExhausted:
            # Only the search session's own output counts as an exhausted search, never claim reproduction.
            outcome = None
        if outcome is None:
            return None
        if isinstance(outcome, Stopped) and outcome.reason is StopReason.CONTEXT_LIMIT:
            return exhausted(name, SearchLimit.CONTEXT_WINDOW)
        if (
            not isinstance(outcome, Returned)
            or outcome.tool != "report"
            or not isinstance(outcome.payload, AdversaryReport)
        ):
            reason = (
                f"{outcome.summary} without report"
                if isinstance(outcome, Stopped)
                else f"Expected report, got {type(outcome).__name__}"
            )
            return AdversaryResult(name, AdversaryStatus.FAILED, (), (reason,))
        report = outcome.payload
        async with asyncio.TaskGroup() as group:
            claims = [
                group.create_task(reproduced(ctx, spec, claim, index)) for index, claim in enumerate(report.findings)
            ]
        results = [claim.result() for claim in claims]
        # A claim that could not be reproduced is incomplete adversary evidence: review cannot accept on it.
        reasons = tuple(
            f"reproducing {finding.claim!r} was inconclusive: {reason}"
            for finding, failure in results
            for reason in failure
        )
        status = AdversaryStatus.FAILED if reasons else AdversaryStatus.REPORTED
        return AdversaryResult(name, status, tuple(finding for finding, _ in results), reasons)


async def validate(ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources) -> Validated:
    submission = workpiece.submission
    if submission is None:
        raise ValueError("Validation requires a submission")
    settings = lowering_settings(config.policy)
    task = submission.task
    try:
        spec = lower_with_settings(task, settings, resources.factories).model_copy(update={"trace": TraceMode.MESSAGES})
    except (ValueError, NotImplementedError) as error:
        return Validated(
            ValidationResult(
                submission.digest,
                (),
                (),
                (),
                lowering_error=str(error),
                lowering_cause=Cause.INVALID_TASK if isinstance(error, InvalidTaskSpec) else Cause.GRADER_INFRA,
            )
        )
    k = config.policy.setting("solver.k", int)
    if k < 1:
        raise ValueError("Validation needs at least one solver trial")
    async with ctx.scope("validate"):
        ctx.observe(Resolved, grader_infra_unavailable)
        roster = config.policy.setting("adversary.roster", tuple)
        async with asyncio.TaskGroup() as group:
            controls = [
                group.create_task(control_result(ctx, spec, index, control))
                for index, control in enumerate(submission.controls)
            ]
            solver = [group.create_task(solver_result(ctx, spec, trial, len(workpiece.evidence))) for trial in range(k)]
            adversary = [
                group.create_task(
                    adversary_result(ctx, workpiece, config, resources, spec, name, roster[:i].count(name))
                )
                for i, name in enumerate(roster)
            ]
        return Validated(
            ValidationResult(
                submission.digest,
                tuple(trial.result() for trial in solver),
                tuple(control.result() for control in controls),
                tuple(result.result() for result in adversary),
            )
        )
