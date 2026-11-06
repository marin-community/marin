# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""

import asyncio
from pathlib import PurePosixPath

from rolloutengine.models import TaskRolloutSpec, TraceMode

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import Resolved
from forge.effects.issues import Hold, Restart
from forge.harness.protocol import Returned, Stopped
from forge.harness.tools.adversary import AdversaryClaim, AdversaryReport, ReportTool
from forge.lowering import host_requirements, lower_with_settings, lowering_settings, requirements
from forge.model import serialization as wire
from forge.model.controls import Control
from forge.model.llm import Message
from forge.model.proposal import render
from forge.model.review import AdversaryFinding, AdversaryResult, ControlResult, TrialResult, ValidationResult
from forge.model.workpiece import Conversation, Workpiece
from forge.resources.bundle import Resources
from forge.resources.env import WriteFile
from forge.resources.grading import GradeCandidate
from forge.resources.rollouts import SolverRollout
from forge.stages.author import hydrate, install_env
from forge.stages.classify import grade_cause, host_failure, restart_cause, trial_cause
from forge.stages.exits import Validated


async def control_result(ctx: Context, spec: TaskRolloutSpec, index: int, control: Control) -> ControlResult:
    try:
        grade = await ctx.emit(GradeCandidate(spec, control.candidate))
    except Restart as error:
        return ControlResult(index, None, False, restart_cause(error))
    cause = grade_cause(grade, spec.task)
    return ControlResult(index, grade, cause is None and control.expected.met_by(grade), cause)


async def solver_result(ctx: Context, spec: TaskRolloutSpec, trial: int, attempt: int) -> TrialResult:
    try:
        result = await ctx.emit(SolverRollout(spec, trial, attempt))
    except Restart as error:
        return TrialResult(trial, None, restart_cause(error))
    return TrialResult(trial, result.grade, trial_cause(result, spec.task))


async def adversary_result(
    ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources, spec: TaskRolloutSpec, name: str
) -> AdversaryResult:
    result = AdversaryResult(name, False, ())
    try:
        result = await run_adversary(ctx, workpiece, config, resources, spec, name)
    except* (Hold, Restart) as error:
        result = AdversaryResult(name, False, (), failure_reasons(error))
    return result


def failure_reasons(error: BaseException) -> tuple[str, ...]:
    """Retain leaf diagnostics when observer scopes group infrastructure failures."""
    if isinstance(error, BaseExceptionGroup):
        return tuple(reason for child in error.exceptions for reason in failure_reasons(child))
    reasons = (f"{type(error).__name__}: {error}",)
    return reasons + (failure_reasons(error.__cause__) if error.__cause__ is not None else ())


async def claim_result(ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim) -> AdversaryFinding:
    try:
        grade = await ctx.emit(GradeCandidate(spec, claim.candidate))
        if grade_cause(grade, spec.task) is None and claim.expected.met_by(grade):
            grade = await ctx.emit(GradeCandidate(spec, claim.candidate, fresh_verifier=True))
    except Restart as error:
        return AdversaryFinding(claim.claim, claim.candidate, claim.expected, None, False, restart_cause(error))
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
            await hydrate(ctx, files)
            task_json = wire.TASK_SPEC.dump_json(spec.task)
            await ctx.emit(
                WriteFile(PurePosixPath(config.policy.setting("author.workdir", str), "task.json"), task_json)
            )
            tools = {**config.tools[name], "report": ReportTool(spec.task)}
            session = Conversation(
                (
                    Message("system", config.policy.prompt("adversary.system").render()),
                    Message(
                        "user",
                        config.policy.prompt("adversary.user").render(
                            task=task_json.decode(), proposal=render(workpiece.proposal), name=name
                        ),
                    ),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            outcome = await config.adversaries[name].run(ctx, session, env, resources.inference_for("adversary"), tools)
            if (
                not isinstance(outcome, Returned)
                or outcome.tool != "report"
                or not isinstance(outcome.payload, AdversaryReport)
            ):
                reason = (
                    f"{outcome.reason} without report"
                    if isinstance(outcome, Stopped)
                    else f"Expected report, got {type(outcome).__name__}"
                )
                return AdversaryResult(name, False, (), (reason,))
            report = outcome.payload
            findings = await asyncio.gather(*(claim_result(ctx, spec, claim) for claim in report.findings))
            return AdversaryResult(name, True, tuple(findings))
    finally:
        await env.close()


async def validate(ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources) -> Validated:
    submission = workpiece.submission
    if submission is None:
        raise ValueError("Validation requires a submission")
    settings = lowering_settings(config.policy)
    task = submission.task
    try:
        spec = lower_with_settings(task, settings, resources.factories).model_copy(update={"trace": TraceMode.MESSAGES})
    except (ValueError, NotImplementedError) as error:
        return Validated(ValidationResult(submission.digest, (), (), (), lowering_error=str(error)))
    k = config.policy.setting("solver.k", int)
    if k < 1:
        raise ValueError("Validation needs at least one solver trial")
    async with ctx.scope("validate"):
        ctx.observe(Resolved, host_failure)
        controls, solver, adversary = await asyncio.gather(
            asyncio.gather(
                *(control_result(ctx, spec, index, control) for index, control in enumerate(submission.controls))
            ),
            asyncio.gather(*(solver_result(ctx, spec, trial, len(workpiece.evidence)) for trial in range(k))),
            asyncio.gather(
                *(
                    adversary_result(ctx, workpiece, config, resources, spec, name)
                    for name in config.policy.setting("adversary.roster", tuple)
                )
            ),
        )
        return Validated(ValidationResult(submission.digest, tuple(solver), tuple(controls), tuple(adversary)))
