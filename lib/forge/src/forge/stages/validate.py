# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""


import asyncio
from pathlib import PurePosixPath

from rolloutengine.models import TaskRolloutSpec, TraceMode
from rolloutengine.validation import InvalidTaskSpec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import resource_bytes

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import Note, Resolved
from forge.effects.issues import Hold, OutputBudgetExhausted, describe_exception
from forge.harness.protocol import Returned, Stopped, StopReason
from forge.harness.tools.adversary import AdversaryClaim, AdversaryReport, ReportTool
from forge.lowering import check_verification, host_requirements, lower_with_settings, lowering_settings, requirements
from forge.model.controls import Control
from forge.model.llm import Message, Role
from forge.model.proposal import render
from forge.model.review import AdversaryFinding, AdversaryResult, Cause, ControlResult, TrialResult, ValidationResult
from forge.model.workpiece import Conversation, Workpiece
from forge.resources.bundle import Resources
from forge.resources.env import WriteFiles
from forge.resources.execution import close_resource
from forge.resources.grading import GradeCandidate
from forge.resources.rollouts import SolverRollout
from forge.stages.author import install_env, sandbox_limits
from forge.stages.classify import grade_cause, grader_infra_unavailable, trial_cause
from forge.stages.exits import Validated
from forge.stages.summary import task_manifest


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
            await ctx.emit(Note(f"Solver {trial}: output budget exhausted on attempt {retry + 1} of 2"))
        if result is not None:
            return TrialResult(trial, result.grade, trial_cause(result, spec.task))
    return TrialResult(trial, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted"))


async def adversary_result(
    ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources, spec: TaskRolloutSpec, name: str
) -> AdversaryResult:
    for retry in range(2):
        result = None
        try:
            result = await run_adversary(ctx, workpiece, config, resources, spec, name)
        except* OutputBudgetExhausted:
            await ctx.emit(Note(f"Adversary {name}: output budget exhausted on attempt {retry + 1} of 2"))
        except* Hold as error:
            result = AdversaryResult(name, False, (), failure_reasons(error))
        if result is not None:
            return result
    return AdversaryResult(name, True, (), ("output budget exhausted",))


def failure_reasons(error: BaseException) -> tuple[str, ...]:
    """Retain leaf diagnostics when observer scopes group infrastructure failures."""
    if isinstance(error, BaseExceptionGroup):
        return tuple(reason for child in error.exceptions for reason in failure_reasons(child))
    reasons = (describe_exception(error),)
    return reasons + (failure_reasons(error.__cause__) if error.__cause__ is not None else ())


async def claim_result(ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim) -> AdversaryFinding:
    grade = await ctx.emit(GradeCandidate(spec, claim.candidate))
    if grade_cause(grade, spec.task) is None and claim.expected.met_by(grade):
        grade = await ctx.emit(GradeCandidate(spec, claim.candidate, fresh_verifier=True))
    cause = grade_cause(grade, spec.task)
    return AdversaryFinding(
        claim.claim, claim.candidate, claim.expected, grade, cause is None and claim.expected.met_by(grade), cause
    )


async def run_adversary(
    ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources, spec: TaskRolloutSpec, name: str
) -> AdversaryResult:
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
                            proposal=render(workpiece.proposal),
                            name=name,
                            workdir=config.policy.setting("author.workdir", str),
                        ),
                    ),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            outcome = await config.adversaries[name].run(ctx, session, resources.inference_for("adversary"), tools)
            if isinstance(outcome, Stopped) and outcome.reason is StopReason.CONTEXT_LIMIT:
                return AdversaryResult(name, True, (), ("adversary exhausted its window without a report",))
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
                return AdversaryResult(name, False, (), (reason,))
            report = outcome.payload
            async with asyncio.TaskGroup() as group:
                findings = [group.create_task(claim_result(ctx, spec, claim)) for claim in report.findings]
            return AdversaryResult(name, True, tuple(finding.result() for finding in findings))
    finally:
        await close_resource(env.close, ctx.emit)


async def validate(ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources) -> Validated:
    submission = workpiece.submission
    if submission is None:
        raise ValueError("Validation requires a submission")
    settings = lowering_settings(config.policy)
    task = submission.task
    try:
        check_verification(task, workpiece.proposal.header.verification, settings)
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
