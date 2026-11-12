# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""


import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import PurePosixPath

from rolloutengine.models import TaskRolloutSpec, TraceMode
from rolloutengine.validation import InvalidTaskSpec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import Note, Resolved
from forge.effects.issues import DecisionModelUnavailable, Hold, OutputBudgetExhausted, describe_exception
from forge.harness.protocol import Returned, Stopped, StopReason
from forge.harness.tools.adversary import AdversaryClaim, AdversaryReport, ReportTool
from forge.harness.tools.reviewer import ReviewedTrial, render_message
from forge.harness.tools.validate import pending_validate
from forge.lowering import check_verification, host_requirements, lower_with_settings, lowering_settings, requirements
from forge.model.content_hash import digest
from forge.model.controls import Control
from forge.model.llm import Message, Role
from forge.model.proposal import render
from forge.model.review import (
    Accept,
    AdversaryFinding,
    AdversaryResult,
    Cause,
    ControlResult,
    Modify,
    Observation,
    ObservedStatus,
    Revise,
    TrialResult,
    ValidationResult,
)
from forge.model.workpiece import Conversation, Workpiece
from forge.resources.bundle import Resources
from forge.resources.env import WriteFiles
from forge.resources.execution import close_resource
from forge.resources.grading import GradeCandidate
from forge.resources.rollouts import SolverRollout
from forge.stages.author import install_env, sandbox_limits
from forge.stages.classify import CAUSE_DECISIONS, grade_cause, grader_infra_unavailable, trial_cause
from forge.stages.exits import Validated
from forge.stages.reviewer import run_reviewer
from forge.stages.summary import excerpt, grade_summary, task_manifest


async def control_result(ctx: Context, spec: TaskRolloutSpec, index: int, control: Control) -> ControlResult:
    grade = await ctx.emit(GradeCandidate(spec, control.candidate))
    cause = grade_cause(grade, spec.task)
    return ControlResult(index, grade, cause is None and control.expected.met_by(grade), cause)


async def solver_result(
    ctx: Context, spec: TaskRolloutSpec, trial: int, attempt: int
) -> tuple[TrialResult, tuple[str, ...]]:
    """A trial's grade, and its transcript (every conversation, one rendered string per message) for the reviewer."""
    for retry in range(2):
        result = None
        try:
            result = await ctx.emit(SolverRollout(spec, trial, attempt))
        except* OutputBudgetExhausted:
            await ctx.emit(Note(f"Solver {trial}: output budget exhausted on attempt {retry + 1} of 2"))
        if result is not None:
            messages = tuple(
                render_message(message)
                for conversation in result.rollout.conversations
                for message in conversation.messages
            )
            return TrialResult(trial, result.grade, trial_cause(result, spec.task)), messages
    return TrialResult(trial, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted")), ()


async def adversary_result(
    ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources, spec: TaskRolloutSpec, name: str
) -> AdversaryResult:
    for retry in range(2):
        result = None
        try:
            result = await run_adversary(ctx, workpiece, config, resources, spec, name)
        except* OutputBudgetExhausted:
            await ctx.emit(Note(f"Adversary {name}: output budget exhausted on attempt {retry + 1} of 2"))
        except* DecisionModelUnavailable:
            raise  # holds the whole workpiece so the driver frees the slot and defers decision_model proposals
        except* Hold as error:
            result = AdversaryResult(name, False, (), failure_reasons(error))
        if result is not None:
            return result
    return AdversaryResult(name, True, (), ("output budget exhausted",))


def failure_reasons(error: BaseException) -> tuple[str, ...]:
    """Retain leaf diagnostics when observer scopes group infrastructure failures; each exception once."""
    reasons: list[str] = []
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
            continue
        reasons.append(describe_exception(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
    return tuple(reasons)


def observation(grade: GradeResult, task: TaskSpec) -> Observation | None:
    """What a replay shows; infrastructure failures show nothing about the grader."""
    cause = grade_cause(grade, task)
    if cause is not None:
        return Observation(ObservedStatus.GRADER_ERROR) if CAUSE_DECISIONS.get(cause) is Modify else None
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return Observation(ObservedStatus.SUBMISSION_FAILURE)
    return Observation(ObservedStatus.GRADED, grade.reward)


def reproduces(claim: AdversaryClaim, grade: GradeResult, task: TaskSpec) -> bool:
    if claim.defect is None:
        return grade_cause(grade, task) is None and claim.expected is not None and claim.expected.met_by(grade)
    shown = observation(grade, task)
    return shown is not None and claim.defect.shown_by(shown)


async def claim_result(
    ctx: Context, spec: TaskRolloutSpec, claim: AdversaryClaim, revision: int = 0
) -> AdversaryFinding:
    grade = await ctx.emit(GradeCandidate(spec, claim.candidate))
    if grade_cause(grade, spec.task) is None and reproduces(claim, grade, spec.task):
        grade = await ctx.emit(GradeCandidate(spec, claim.candidate, fresh_verifier=True))
    return AdversaryFinding(
        claim.claim,
        claim.candidate,
        claim.expected,
        grade,
        reproduces(claim, grade, spec.task),
        grade_cause(grade, spec.task),
        claim.defect,
        revision,
    )


def counted(finding: AdversaryFinding) -> bool:
    """Whether review treats the finding as a defect (see ``rules.assess``)."""
    return finding.met or CAUSE_DECISIONS.get(finding.cause) is Modify


def replay_feedback(missed: Sequence[AdversaryFinding], remaining: int) -> str:
    """Tell the adversary what the replay saw for each finding that did not reproduce."""
    lines = [f"Validation replayed your report against the original task. {len(missed)} finding(s) did not reproduce:"]
    for index, finding in enumerate(missed):
        claim = finding.defect
        reported = (
            f"defect {claim.defect}, grader_observed {claim.observed.status} reward {claim.observed.reward}, "
            f"deserved_reward {claim.deserved_reward}"
            if claim is not None
            else f"expected {finding.expected}"
        )
        replay = {key: value for key, value in grade_summary(finding.grade).items() if value not in ("", None)}
        if finding.cause is not None:
            replay["cause"] = finding.cause
        lines += [
            f"{index + 1}. Claim: {excerpt(finding.claim)}",
            f"   You reported: {reported}.",
            f"   The replay observed: {json.dumps(replay, default=str)}.",
        ]
    lines.append(
        "The replay failed to reproduce these findings, so they do not count. You may call report again with a "
        "revised submission (fix the candidate, or correct the fields so grader_observed_* says what the grader "
        "actually returns and deserved_reward says what the candidate should get), or submit an entirely new defect. "
        "A finding still has to show the grade is WRONG and say why; restating a correct grade is not a finding. "
        "Findings that already reproduced are kept, so do not repeat them. If you cannot demonstrate a defect, call "
        "report with an empty findings list. "
        + (f"After this you have {remaining} more revision(s)." if remaining else "This is your last revision.")
    )
    return "\n".join(lines)


def answer_report(session: Conversation, content: str) -> Conversation:
    """Answer the session's pending report call, so the adversary continues in the same conversation."""
    calls = session.messages[-1].tool_calls if session.messages else ()
    report = next((call for call in calls if call.name == "report"), None)
    message = Message(Role.TOOL, content, tool_call_id=report.id) if report is not None else Message(Role.USER, content)
    return replace(session, messages=(*session.messages, message))


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
            graded: dict[str, Observation] = {}

            async def remember(event: Resolved[object]) -> None:
                # Every grade of a candidate in this session, the adversary's own and the replays alike.
                if isinstance(event.request, GradeCandidate) and isinstance(event.value, GradeResult):
                    if (shown := observation(event.value, spec.task)) is not None:
                        graded[digest(event.request.candidate)] = shown

            ctx.observe(Resolved, remember)
            tools = {**config.tools[name], "report": ReportTool(spec.task, graded)}
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
            harness, client = config.adversaries[name], resources.inference_for("adversary")
            outcome = await harness.run(ctx, session, client, tools)
            findings: tuple[AdversaryFinding, ...] = ()
            revisions = config.policy.setting("adversary.max_revisions", int)
            for revision in range(revisions + 1):
                if isinstance(outcome, Stopped) and outcome.reason is StopReason.CONTEXT_LIMIT:
                    reason = "adversary exhausted its window without a report"
                    if revision:
                        await ctx.emit(Note(f"Adversary {name}: revision {revision}: {reason}"))
                        break
                    return AdversaryResult(name, True, (), (reason,))
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
                    if revision:
                        await ctx.emit(Note(f"Adversary {name}: revision {revision}: {reason}"))
                        break
                    return AdversaryResult(name, False, (), (reason,))
                report = outcome.payload
                async with asyncio.TaskGroup() as group:
                    replays = [group.create_task(claim_result(ctx, spec, claim, revision)) for claim in report.findings]
                replayed = tuple(replay.result() for replay in replays)
                findings += replayed
                missed = tuple(finding for finding in replayed if not counted(finding))
                if not missed or revision == revisions:
                    break
                await ctx.emit(
                    Note(f"Adversary {name}: {len(missed)} finding(s) did not reproduce; revision {revision + 1}")
                )
                feedback = replay_feedback(missed, revisions - revision - 1)
                outcome = await harness.run(ctx, answer_report(outcome.session, feedback), client, tools)
            return AdversaryResult(name, True, findings)
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
        result = ValidationResult(
            submission.digest,
            tuple(trial.result()[0] for trial in solver),
            tuple(control.result() for control in controls),
            tuple(result.result() for result in adversary),
        )
        if reviewable(workpiece, config, result):
            trials = tuple(
                ReviewedTrial(json.dumps({"cause": trial.cause, **grade_summary(trial.grade)}, default=str), messages)
                for trial, messages in (handle.result() for handle in solver)
            )
            result = replace(
                result, reviewer=await run_reviewer(ctx, workpiece, config, resources, task, result, trials)
            )
        return Validated(result)


def reviewable(workpiece: Workpiece, config: ForgeConfig, result: ValidationResult) -> bool:
    """Whether review would accept ``result`` as it stands while reviewer rounds remain.

    A tool validation (arm T) is skipped: the author's later submission is the one the reviewer reads.
    """
    if workpiece.session is not None and pending_validate(workpiece.session) is not None:
        return False
    rounds = sum(isinstance(decision, Revise) for decision in workpiece.decisions)
    if rounds >= config.policy.setting("reviewer.max_rounds", int):
        return False
    decision = config.review(workpiece.with_(evidence=(*workpiece.evidence, result)), config.band, config.budget)
    return isinstance(decision, Accept) and not decision.revert
