# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""

import asyncio
from pathlib import PurePosixPath

from rolloutengine.models import TaskRolloutSpec, TraceMode

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import Resolved
from forge.effects.issues import Restart
from forge.harness.protocol import Returned
from forge.harness.tools.adversary import AdversaryReport
from forge.lowering import SHELL_CAPABILITY, lower, lowering_settings, requirements
from forge.model import serialization as wire
from forge.model.controls import Control
from forge.model.llm import Message
from forge.model.proposal import render
from forge.model.review import AdversaryResult, ControlResult, TrialResult, ValidationResult
from forge.model.workpiece import Conversation, Workpiece
from forge.resources.bundle import Resources
from forge.resources.env import WriteFile
from forge.resources.grading import GradeCandidate
from forge.resources.rollouts import SolverRollout
from forge.resources.sandbox.factories import grading_environment
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
    requested = spec.task.environment_requirements if spec.task_machine is not None else requirements(image=None)
    try:
        env = await resources.envs.create(requested)
    except Restart:
        return AdversaryResult(name, False, ())
    try:
        async with ctx.scope(f"adversary:{name}"):
            install_env(ctx, env)
            await hydrate(ctx, (*spec.task.resources.all, *spec.task.resources.worker))
            task_json = wire.TASK_SPEC.dump_json(spec.task)
            await ctx.emit(
                WriteFile(PurePosixPath(config.policy.setting("author.workdir", str), "task.json"), task_json)
            )
            tools = config.tools[name]
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
            outcome = await config.adversaries[name].run(ctx, session, env, resources.inference, tools)
            if (
                not isinstance(outcome, Returned)
                or outcome.tool != "report"
                or not isinstance(outcome.payload, AdversaryReport)
            ):
                return AdversaryResult(name, False, ())
            report = outcome.payload
            grades = await asyncio.gather(
                *(asyncio.ensure_future(ctx.emit(GradeCandidate(spec, candidate))) for candidate in report.candidates)
            )
            return AdversaryResult(name, True, report.findings, tuple(grades))
    except Restart:
        return AdversaryResult(name, False, ())
    finally:
        await env.close()


async def validate(ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources) -> Validated:
    submission = workpiece.submission
    if submission is None:
        raise ValueError("Validation requires a submission")
    settings = lowering_settings(config.policy)
    task = submission.task
    try:
        spec = lower(
            task,
            host=settings.host,
            task_machine=(
                settings.task_machine if SHELL_CAPABILITY in task.environment_requirements.capabilities else None
            ),
            verifier_machine=settings.verifier_machine if grading_environment(task) is not None else None,
            limits=settings.limits,
            factories=resources.factories,
        ).model_copy(update={"trace": TraceMode.MESSAGES})
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
