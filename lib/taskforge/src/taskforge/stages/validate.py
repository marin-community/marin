# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""


import asyncio
from pathlib import PurePosixPath

from rolloutengine.models import TaskRolloutSpec, TraceMode
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note, Resolved
from taskforge.effects.issues import EnvironmentSetupFailed, Hold, OutputBudgetExhausted, describe_exception
from taskforge.harness.protocol import Returned, Stopped, StopReason
from taskforge.harness.tools.adversary import AdversaryClaim, AdversaryReport, ReportTool
from taskforge.lowering import (
    InvalidTaskSpec,
    host_requirements,
    lower_with_settings,
    lowering_settings,
    requirements,
)
from taskforge.model.controls import Control
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import render
from taskforge.model.review import (
    AdversaryFinding,
    AdversaryResult,
    AdversaryStatus,
    Cause,
    ControlResult,
    SearchLimit,
    TrialResult,
    ValidationResult,
)
from taskforge.model.workpiece import Conversation, Workpiece
from taskforge.resources.bundle import Resources
from taskforge.resources.env import WriteFiles
from taskforge.resources.execution import close_resource
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import install_env, sandbox_limits
from taskforge.stages.classify import grade_cause, grader_infra_unavailable, trial_cause
from taskforge.stages.exits import Validated
from taskforge.stages.summary import task_manifest

NO_PROPOSAL = "None: the task came from another source without one."


def setup_failure(error: BaseExceptionGroup[EnvironmentSetupFailed]) -> GradeResult:
    """The grade of an attempt whose task environment's setup failed: an invalid task, returned to the author."""
    reasons = "; ".join(str(issue) for issue in error.exceptions)
    return GradeResult(Outcome.INVALID_TASK, None, f"Task environment setup failed: {reasons}")


async def graded(ctx: Context, event: GradeCandidate) -> GradeResult:
    grade: GradeResult | None = None
    try:
        grade = await ctx.emit(event)
    except* EnvironmentSetupFailed as error:
        grade = setup_failure(error)
    assert grade is not None
    return grade


async def control_result(ctx: Context, spec: TaskRolloutSpec, index: int, control: Control) -> ControlResult:
    grade = await graded(ctx, GradeCandidate(spec, control.candidate))
    cause = grade_cause(grade, spec.task)
    return ControlResult(index, grade, cause is None and control.expected.met_by(grade), cause)


async def solver_result(ctx: Context, spec: TaskRolloutSpec, trial: int, attempt: int) -> TrialResult:
    for retry in range(2):
        result = None
        setup: GradeResult | None = None
        try:
            result = await ctx.emit(SolverRollout(spec, trial, attempt))
        except* OutputBudgetExhausted:
            await ctx.emit(Note(f"Solver {trial}: output budget exhausted on attempt {retry + 1} of 2"))
        except* EnvironmentSetupFailed as error:
            setup = setup_failure(error)
        if setup is not None:
            return TrialResult(trial, setup, Cause.INVALID_TASK)
        if result is not None:
            return TrialResult(trial, result.grade, trial_cause(result, spec.task))
    return TrialResult(trial, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted"))


async def adversary_result(
    ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources, spec: TaskRolloutSpec, name: str
) -> AdversaryResult:
    for retry in range(2):
        result = None
        try:
            result = await run_adversary(ctx, workpiece, config, resources, spec, name)
        except* Hold as error:
            result = AdversaryResult(name, AdversaryStatus.FAILED, (), failure_reasons(error))
        except* EnvironmentSetupFailed as error:
            reasons = tuple(f"task environment setup failed: {issue}" for issue in error.exceptions)
            result = AdversaryResult(name, AdversaryStatus.UNSTARTED, (), reasons)
        if result is not None:
            return result
        await ctx.emit(Note(f"Adversary {name}: output budget exhausted on attempt {retry + 1} of 2"))
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


async def claim_result(ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim) -> AdversaryFinding:
    grade = await graded(ctx, GradeCandidate(spec, claim.candidate))
    if grade_cause(grade, spec.task) is None and claim.expected.met_by(grade):
        grade = await graded(ctx, GradeCandidate(spec, claim.candidate, fresh_verifier=True))
    cause = grade_cause(grade, spec.task)
    return AdversaryFinding(
        claim.claim, claim.candidate, claim.expected, grade, cause is None and claim.expected.met_by(grade), cause
    )


async def reproduced(
    ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim, index: int
) -> tuple[AdversaryFinding, tuple[str, ...]]:
    """Reproduce one claim; an inconclusive reproduction keeps the claim and candidate ungraded, with why.

    Its own scope collects observer holds on its grades, so a sibling claim's grade survives them.
    """
    finding, reasons = AdversaryFinding(claim.claim, claim.candidate, claim.expected, None, False), ()
    try:
        async with ctx.scope(f"claim:{index}"):
            graded = await claim_result(ctx, spec, claim)
        finding = graded
    except* (OutputBudgetExhausted, Hold) as error:
        reasons = failure_reasons(error)
    return finding, reasons


async def run_adversary(
    ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources, spec: TaskRolloutSpec, name: str
) -> AdversaryResult | None:
    """One adversary search and the reproduction of its claims; ``None`` when the search ran out of output."""
    requested = (
        spec.task.environment_requirements
        if spec.task_machine is not None
        else requirements(
            image=config.policy.setting("adversary.image", str) or None,
            workdir=config.policy.setting("author.workdir", str),
        )
    )
    files = (*spec.task.resources.all, *spec.task.resources.worker)
    env = await resources.envs.create(host_requirements(requested))
    try:
        async with ctx.scope(f"adversary:{name}"):
            install_env(ctx, env)
            task_json = spec.task.model_dump_json().encode()
            await ctx.emit(
                WriteFiles(
                    (
                        *((PurePosixPath("/", file.path), resource_bytes(file)) for file in files),
                        (PurePosixPath(config.policy.setting("author.workdir", str), "task.json"), task_json),
                    )
                )
            )
            tools = {**config.tools[name], "report": ReportTool(spec.task)}
            session = Conversation(
                (
                    Message(
                        Role.SYSTEM,
                        config.policy.prompt("adversary.system").render(sandbox_limits=sandbox_limits(config.policy)),
                    ),
                    Message(
                        Role.USER,
                        config.policy.prompt("adversary.user").render(
                            task=task_manifest(spec.task),
                            proposal=NO_PROPOSAL if workpiece.proposal is None else render(workpiece.proposal),
                            name=name,
                            workdir=config.policy.setting("author.workdir", str),
                        ),
                    ),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            try:
                outcome = await config.adversaries[name].run(ctx, session, resources.inference_for("adversary"), tools)
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
    finally:
        await close_resource(env.close, ctx.emit)


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
        async with asyncio.TaskGroup() as group:
            controls = [
                group.create_task(control_result(ctx, spec, index, control))
                for index, control in enumerate(submission.controls)
            ]
            solver = [group.create_task(solver_result(ctx, spec, trial, len(workpiece.evidence))) for trial in range(k)]
            adversary = [
                group.create_task(adversary_result(ctx, workpiece, config, resources, spec, name))
                for name in config.policy.setting("adversary.roster", tuple)
            ]
        return Validated(
            ValidationResult(
                submission.digest,
                tuple(trial.result() for trial in solver),
                tuple(control.result() for control in controls),
                tuple(result.result() for result in adversary),
            )
        )
