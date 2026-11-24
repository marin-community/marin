# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build validation evidence through concurrent grading, solver and adversary effects."""


import asyncio
import itertools
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath
from typing import cast

from rolloutengine.models import GradedRollout, TaskRolloutSpec, TraceMode
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note, Resolved
from taskforge.effects.issues import DecisionModelUnavailable, Hold, OutputBudgetExhausted, Restart, describe_exception
from taskforge.effects.log import Billing
from taskforge.harness.protocol import Returned, Stopped, StopReason
from taskforge.harness.role import Answer, Files, Fresh, Visit, sitting
from taskforge.harness.tools.adversary import AdversaryClaim, AdversaryReport, ReportTool
from taskforge.harness.tools.evidence import FinalReportTool
from taskforge.lowering import (
    InvalidTaskSpec,
    lower_with_settings,
    lowering_settings,
    requirements,
)
from taskforge.model.content_hash import digest
from taskforge.model.controls import Control
from taskforge.model.llm import Role
from taskforge.model.proposal import render
from taskforge.model.review import (
    Accept,
    AdversaryFinding,
    AdversaryResult,
    AdversaryStatus,
    Cause,
    Continuation,
    ControlResult,
    Incomplete,
    Modify,
    Observation,
    ObservedStatus,
    Reject,
    Revise,
    SearchEnd,
    SearchLimit,
    SolveReview,
    TrialOutcome,
    TrialResult,
    ValidationResult,
)
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Conversation, Submission, Workpiece
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import sandbox_limits
from taskforge.stages.classify import (
    CAUSE_DECISIONS,
    grade_cause,
    grader_infra_unavailable,
    trial_cause,
)
from taskforge.stages.exits import Validated
from taskforge.stages.rules import actionable, assess, earlier_findings, shipped
from taskforge.stages.solve_review import earlier_reports, evidence, shipped_notes, solve_review
from taskforge.stages.summary import task_manifest


async def control_result(ctx: Context, spec: TaskRolloutSpec, index: int, control: Control) -> ControlResult:
    grade = await ctx.emit(GradeCandidate(spec, control.candidate))
    cause = grade_cause(grade, spec.task)
    return ControlResult(index, grade, cause is None and control.expected.met_by(grade), cause)


async def solver_result(
    ctx: Context, spec: TaskRolloutSpec, trial: int, attempt: int, rollouts: dict[int, GradedRollout] | None = None
) -> TrialResult:
    """One slot's ``SolverRollout``, kept in ``rollouts`` for the solve-review: a slot-local hold is recorded missing
    evidence, and a rollout that exhausts its output budget twice is recorded as a submission failure; any other hold
    or restart propagates."""
    for retry in range(2):
        try:
            result = await ctx.emit(SolverRollout(spec, trial, attempt))
        except OutputBudgetExhausted:
            ctx.notify(Note(f"Solver {trial}: output budget exhausted on attempt {retry + 1} of 2"))
            continue
        receipt = ctx.receipt(SolverRollout(spec, trial, attempt))  # this outcome's own row, read before any other runs
        if rollouts is not None:
            rollouts[trial] = result
        if (cause := trial_cause(result, spec.task)) is Cause.HELD:
            return TrialResult(trial, None, cause, result.grade.error or "", receipt=receipt)
        return TrialResult(trial, result.grade, cause, receipt=receipt)
    return TrialResult(trial, GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "output budget exhausted"))


async def solver_batch(
    ctx: Context,
    spec: TaskRolloutSpec,
    phase: str,
    attempt: int,
    occupied: Sequence[int],
    eligible: int,
    target: int,
    *,
    concurrency: int,
    heartbeat: float,
    max_missing: int = 0,
    stopped: Callable[[], bool] = lambda: False,
    rollouts: dict[int, GradedRollout] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[TrialResult, ...]:
    """Add ``target - eligible`` slots to ``phase``, each its own ``SolverRollout`` of ``attempt`` numbered past every
    ``occupied`` trial index, eligible or not: fresh evidence takes a fresh identity, so no two phases share one,
    while a retry or replay of one unresolved slot keeps its own.

    ``concurrency`` slots run at a time and a ``Note`` lists the running ones every ``heartbeat`` seconds; neither
    enters a replay key. A slot that restarts raises once every sibling has settled, so restarting the stage
    re-executes only it; more than ``max_missing`` held slots hold the batch instead. A decision-model outage defers
    the batch the same way, and any other hold a slot raises holds it, each once every admitted sibling has settled.
    Once a slot waits for the decision model or holds, or ``stopped()`` says a validation sibling holds, no further
    slot starts, so no rollout is bought to wait; the batch then returns short only beside that sibling.
    """
    first = max(occupied, default=-1) + 1
    gate, running = asyncio.Semaphore(concurrency), dict[int, float]()
    outages: list[DecisionModelUnavailable] = []
    holds: list[Hold] = []

    async def slot(trial: int) -> TrialResult | Restart | DecisionModelUnavailable | None:
        async with gate:
            if outages or holds or stopped():
                return None
            running[trial] = clock()
            try:
                return await solver_result(ctx, spec, trial, attempt, rollouts)
            except Restart as error:
                return error
            except DecisionModelUnavailable as error:
                outages.append(error)
                return error
            except Hold as error:  # such as a decision model refusing a read: it must not cancel a sibling's evidence
                holds.append(error)
                return None
            finally:
                del running[trial]

    async def beat() -> None:
        while True:
            await sleep(heartbeat)
            now = clock()
            listed = "".join(f" [trial {trial} {now - started:.0f}s]" for trial, started in running.items())
            ctx.notify(Note(f"Solver {phase} attempt {attempt}: {len(running)} running{listed}"))

    ticker = asyncio.create_task(beat())
    try:
        async with asyncio.TaskGroup() as group:
            slots = [group.create_task(slot(trial)) for trial in range(first, first + target - eligible)]
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
    results = [slot.result() for slot in slots]
    unfinished = [result for result in results if isinstance(result, Restart | DecisionModelUnavailable)]
    held = refusal([r for r in results if isinstance(r, TrialResult)], max_missing)
    if holds:
        raise BaseExceptionGroup(f"Solver {phase} slots held", [*holds, *([held] if held else [])])
    if unfinished and held:
        raise held  # what review would decide, before anything unfinished is bought again
    if unfinished:
        raise BaseExceptionGroup(f"Solver {phase} slots did not finish", unfinished)
    return tuple(result for result in results if isinstance(result, TrialResult))


def refusal(trials: Sequence[TrialResult], max_missing: int) -> Hold | None:
    """More than ``max_missing`` held slots: a human hold, whatever else is unfinished."""
    held = [f"solver trial {trial.trial}: {trial.detail}" for trial in trials if trial.cause is Cause.HELD]
    return Hold("; ".join(held)) if len(held) > max_missing else None


async def settled[T](work: Awaitable[T], holds: list[BaseException]) -> T | None:
    """``work``'s result, or None with its hold kept, so that no hold cancels a sibling's evidence."""
    try:
        return await work
    except* Hold as held:
        holds.append(held)
    return None


async def raise_holds(
    ctx: Context,
    what: str,
    holds: list[BaseException],
    trials: Sequence[TrialResult] | None,
    max_missing: int,
    beside: Sequence[BaseException] = (),
) -> None:
    """Once a wave's work has settled into ``holds``: drain its grade observers into them too, then raise every hold
    at once, with what else stops the wave (``beside``) and the batch's own refusal, so an outage defers only a wave
    whose every other disposition is settled evidence."""
    await settled(ctx.check(), holds)
    if holds:
        holds += [*beside, *([held] if (held := refusal(trials or (), max_missing)) else [])]
        raise BaseExceptionGroup(f"{what} holds", holds)


def kept[T](task: asyncio.Task[T | None]) -> T:
    result = task.result()
    assert result is not None, "only a hold leaves no result, and a hold raised"
    return result


async def adversary_result(
    ctx: Context,
    workpiece: Workpiece,
    config: TaskforgeConfig,
    resources: Resources,
    spec: TaskRolloutSpec,
    name: str,
    occurrence: int,
) -> AdversaryResult:
    """One roster entry, ``occurrence`` among entries with its name; each search runs on its own machine."""
    visit = f"validate/{workpiece.round}/{len(workpiece.evidence)}"
    owner = ""
    for retry in range(2):
        result = None
        try:
            owner = f"{visit}/adversary:{name}/{occurrence}/{retry}"
            result = await run_adversary(ctx, workpiece, config, resources, spec, name, owner)
        except* DecisionModelUnavailable:
            raise  # defers the workpiece; it is no failed search
        except* Hold as error:
            held = Continuation(owner, SearchEnd.HELD)
            result = AdversaryResult(name, AdversaryStatus.FAILED, (), failure_reasons(error), continuation=held)
        if result is not None:
            return result
        ctx.notify(Note(f"Adversary {name}: output budget exhausted on attempt {retry + 1} of 2"))
    return exhausted(name, SearchLimit.OUTPUT_BUDGET, Continuation(owner, SearchEnd.BUDGET_EXHAUSTED))


REPLAY_CLAIM_CHARS = 600
"""How much of a claim the replay feedback quotes back to its adversary."""


def no_finding(how: str) -> str:
    """The one wording for a search that ended without a reproduced finding: budget spent, never a validation."""
    return f"ran out of search budget without a finding ({how}); this is not a validation of the task"


def exhausted(name: str, limit: SearchLimit, continuation: Continuation) -> AdversaryResult:
    reason = no_finding(f"its {limit} ran out before it reported")
    return AdversaryResult(name, AdversaryStatus.EXHAUSTED, (), (reason,), limit, continuation)


def stopped_at(owner: str, end: SearchEnd, ended: Exit | Conversation | None) -> Continuation:
    """Where a session ended: at its exit, or at the conversation it held when a command raised."""
    session = ended.session if isinstance(ended, Returned | Stopped) else ended
    if not isinstance(session, Conversation):
        return Continuation(owner, end)
    return Continuation(owner, end, digest(session), sum(message.role == Role.ASSISTANT for message in session.messages))


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
    try:
        async with ctx.scope(f"claim:{index}"):  # its exit may still raise, and then the claim stays ungraded
            return await claim_result(ctx, spec, claim), ()
    except* DecisionModelUnavailable:
        raise  # defers the workpiece; the reproduction is not inconclusive
    except* (OutputBudgetExhausted, Hold) as error:
        reasons = failure_reasons(error)
    return AdversaryFinding(claim.claim, claim.candidate, claim.defect, None, False), reasons


def report_of(outcome: object) -> Returned[AdversaryReport] | None:
    if isinstance(outcome, Returned) and outcome.tool == "report" and isinstance(outcome.payload, AdversaryReport):
        return cast(Returned[AdversaryReport], outcome)
    return None


def replay_feedback(missed: Sequence[AdversaryFinding], remaining: int) -> str:
    """Tell the adversary what the replay saw for each finding that did not reproduce."""
    lines = [f"Validation replayed your report against the original task. {len(missed)} finding(s) did not reproduce:"]
    for index, finding in enumerate(missed):
        claim, grade = finding.defect, finding.grade
        replay = {} if grade is None else {"status": grade.status, "reward": grade.reward, "error": grade.error}
        replay = {key: value for key, value in {**replay, "cause": finding.cause}.items() if value not in ("", None)}
        lines += [
            f"{index + 1}. Claim: {finding.claim[:REPLAY_CLAIM_CHARS]}",
            f"   You reported: defect {claim.defect}, grader_observed {claim.observed.status} reward "
            f"{claim.observed.reward}, deserved_reward {claim.deserved_reward}.",
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

    def files() -> Files:
        shared = (*spec.task.resources.all, *spec.task.resources.worker)
        task = (PurePosixPath(workdir, "task.json"), spec.task.model_dump_json().encode())
        return (*((PurePosixPath("/", file.path), resource_bytes(file)) for file in shared), task)

    graded: dict[str, Observation] = {}

    async def remember(event: Resolved[object]) -> None:
        # Every grade of a candidate in this visit, the adversary's own and the replays alike.
        if isinstance(event.request, GradeCandidate) and isinstance(event.value, GradeResult):
            if (shown := observation(event.value, spec.task)) is not None:
                graded[digest(event.request.candidate)] = shown

    tools = {"report": ReportTool(spec.task, graded)}
    visit = Visit(f"adversary:{name}", machine, files, tools, bills=Billing.ADVERSARY)
    async with sitting(ctx, config.adversary, visit, resources) as session:
        ctx.observe(Resolved, remember)
        brief = f"adversary.brief.{name}"  # without one, the name is the strategy, as wiring allows for {name} prompts
        opening = Fresh(
            config.policy.prompt("adversary.system").render(sandbox_limits=sandbox_limits(config.policy)),
            config.policy.prompt("adversary.user").render(
                task=task_manifest(spec.task),
                proposal=render(workpiece.proposal),
                name=name,
                brief=config.policy.prompt(brief).template if brief in config.policy.entries else name,
                workdir=workdir,
            ),
        )
        try:
            outcome = await session(opening)
        except* OutputBudgetExhausted:
            # Only the search session's own output counts as an exhausted search, never claim reproduction.
            outcome = None
        if outcome is None:
            return None
        if isinstance(outcome, Stopped) and outcome.reason is StopReason.CONTEXT_LIMIT:
            return exhausted(name, SearchLimit.CONTEXT_WINDOW, stopped_at(owner, SearchEnd.BUDGET_EXHAUSTED, outcome))
        if (report := report_of(outcome)) is None:
            reason = (
                f"{outcome.summary} without report"
                if isinstance(outcome, Stopped)
                else f"Expected report, got {type(outcome).__name__}"
            )
            return AdversaryResult(
                name, AdversaryStatus.FAILED, (), (reason,), continuation=stopped_at(owner, SearchEnd.HELD, outcome)
            )
        revisions = config.policy.setting("adversary.max_revisions", int)
        replays: list[tuple[AdversaryFinding, tuple[str, ...]]] = []
        end, how, limit = SearchEnd.REPORTED, "its session ended with a report and no reproduced finding", None
        ended: Exit | Conversation | None = report
        for revision in itertools.count(1):  # the first report is reproduced whatever the revision budget
            async with asyncio.TaskGroup() as group:
                claims = [
                    group.create_task(reproduced(ctx, spec, claim, len(replays) + index))
                    for index, claim in enumerate(report.payload.findings)
                ]
            replays += (results := [claim.result() for claim in claims])
            missed = tuple(finding for finding, failure in results if not failure and not actionable(finding))
            if not missed or revision > revisions:
                break
            ctx.notify(Note(f"Adversary {name}: {len(missed)} finding(s) did not reproduce; revision {revision}"))
            try:
                ended = await session(Answer(report.session, replay_feedback(missed, revisions - revision)))
            except* OutputBudgetExhausted as error:  # ends at the conversation the exhausted command left
                limit = SearchLimit.OUTPUT_BUDGET
                ended = next((e.session for e in error.exceptions if isinstance(e, OutputBudgetExhausted)), None)
            except* Hold as error:
                # A revision is extra search: its hold leaves a finding that already reproduced standing.
                if not any(actionable(finding) for finding, _ in replays):
                    raise
                ctx.notify(Note(f"Adversary {name}: revision {revision} held: {'; '.join(failure_reasons(error))}"))
                ended = None
            if (revised := report_of(ended)) is None:  # the last report stands
                how = f"its revision {revision} ended without a report"
                break
            report = revised
        if isinstance(ended, Stopped) and ended.reason is StopReason.CONTEXT_LIMIT:
            limit = SearchLimit.CONTEXT_WINDOW
        if limit is not None:  # a revision ran out of search budget, which consumes the remaining revisions
            end, how = SearchEnd.BUDGET_EXHAUSTED, f"its {limit} ran out during revision {revision}"
        findings = tuple(finding for finding, _ in replays)
        # A claim that could not be reproduced is incomplete adversary evidence: review cannot accept on it.
        reasons = tuple(
            f"reproducing {f.claim!r} was inconclusive: {reason}" for f, failure in replays for reason in failure
        )
        status = AdversaryStatus.FAILED if reasons else AdversaryStatus.REPORTED
        if not reasons and not any(map(actionable, findings)):
            reasons = (no_finding(how),)
        return AdversaryResult(name, status, findings, reasons, continuation=stopped_at(owner, end, ended))


async def reviewed(
    ctx: Context,
    workpiece: Workpiece,
    config: TaskforgeConfig,
    resources: Resources,
    solver: Awaitable[tuple[TrialResult, ...] | None],
    controls: Sequence[Awaitable[ControlResult | None]],
    rollouts: dict[int, GradedRollout],
    holds: list[BaseException],
) -> SolveReview | None:
    """The round's solve-review, a sibling of the adversaries once the trials and controls settle; none when a sibling
    holds first or review would not decide on them (incomplete, rejected)."""
    trials, graded = await solver, [await control for control in controls]
    if holds or trials is None or any(control is None for control in graded) or workpiece.submission is None:
        return None
    probe = ValidationResult(workpiece.submission.digest, trials, tuple(filter(None, graded)), ())
    if isinstance(assess(probe, config.band, config.budget.max_missing_trials), Incomplete | Reject):
        return None
    pairs = [(trial, rollouts.get(trial.trial)) for trial in trials]
    return await solve_review(ctx, config, resources, workpiece.proposal, workpiece.submission, pairs)


def solver_spec(submission: Submission, config: TaskforgeConfig, resources: Resources) -> TaskRolloutSpec:
    task = lower_with_settings(submission.task, lowering_settings(config.policy), resources.factories)
    return task.model_copy(update={"trace": TraceMode.MESSAGES})


async def validate(ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources) -> Validated:
    submission = workpiece.submission
    if submission is None:
        raise ValueError("Validation requires a submission")
    try:
        spec = solver_spec(submission, config, resources)
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
    async with ctx.scope("validate", bills=Billing.SOLVER):
        ctx.observe(Resolved, grader_infra_unavailable)
        roster = config.policy.setting("adversary.roster", tuple)
        holds: list[BaseException] = []  # every sibling settles before a hold or an outage stops the workpiece
        rollouts: dict[int, GradedRollout] = {}
        async with asyncio.TaskGroup() as group:

            def start[T](work: Awaitable[T]) -> asyncio.Task[T | None]:
                return group.create_task(settled(work, holds))

            controls = [start(control_result(ctx, spec, i, control)) for i, control in enumerate(submission.controls)]
            solver = start(
                solver_batch(
                    ctx,
                    spec,
                    "validation",
                    len(workpiece.evidence),
                    (),
                    0,
                    k,
                    concurrency=config.policy.setting("solver.concurrency", int),
                    heartbeat=config.policy.setting("solver.heartbeat_interval", float),
                    max_missing=config.budget.max_missing_trials,
                    stopped=lambda: bool(holds),
                    rollouts=rollouts,
                )
            )
            adversary = [
                start(adversary_result(ctx, workpiece, config, resources, spec, name, roster[:i].count(name)))
                for i, name in enumerate(roster)
            ]
            review = start(reviewed(ctx, workpiece, config, resources, solver, controls, rollouts, holds))
        failed = [a for a in map(asyncio.Task.result, adversary) if a and a.status is AdversaryStatus.FAILED]
        beside = [Hold(f"{a.name}: {'; '.join(a.reasons)}") for a in failed]
        await raise_holds(ctx, "Validation", holds, solver.result(), config.budget.max_missing_trials, beside)
        controls_done, adversaries_done = tuple(map(kept, controls)), tuple(map(kept, adversary))
        result = ValidationResult(submission.digest, kept(solver), controls_done, adversaries_done)
        return Validated(replace(result, review=review.result()))


async def confirm(ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources) -> Validated:
    """The confirmation step of the pending acceptance: the round it ships (``shipped``, a restore's included) keeps its
    scored trials, read from the journal, never bought again; the existing batch adds slots past every index that
    round used, up to ``solver.k + solver.confirmation_k`` under a fresh attempt, so no validation slot's identity
    changes; then the final review reads all of them. Controls and adversaries are not run again. The batch and its
    grade observers settle as validation's do (``raise_holds``), so an outage beside another hold is not a pure one.
    """
    accepted, submission = workpiece.decisions[-1], workpiece.submission
    assert isinstance(accepted, Accept) and submission is not None, "confirmation follows an acceptance"
    index = shipped(accepted, workpiece)
    round_, policy, max_missing = workpiece.evidence[index], config.policy, config.budget.max_missing_trials
    spec = solver_spec(submission, config, resources)
    base = tuple(trial for trial in round_.solver if trial.outcome is not TrialOutcome.UNSCORED)
    async with ctx.scope("confirm", bills=Billing.CONFIRMATION):
        ctx.observe(Resolved, grader_infra_unavailable)
        # Each base trial's own transcript, at its receipt; a trial without one stays without.
        rollouts = {
            t.trial: ctx.recorded_at(t.receipt, SolverRollout(spec, t.trial, index))
            for t in base
            if t.receipt is not None
        }
        holds: list[BaseException] = []
        batch = solver_batch(
            ctx,
            spec,
            "confirmation",
            len(workpiece.evidence),
            [trial.trial for trial in round_.solver],
            len(base),
            policy.setting("solver.k", int) + policy.setting("solver.confirmation_k", int),
            concurrency=policy.setting("solver.concurrency", int),
            heartbeat=policy.setting("solver.heartbeat_interval", float),
            max_missing=max_missing,
            rollouts=rollouts,
        )
        added = await settled(batch, holds)
        await raise_holds(ctx, "Confirmation", holds, added, max_missing)
        assert added is not None, "only a hold leaves no result, and a hold raised"
        if held := refusal(added, max_missing):
            raise held
        trials = (*base, *added)
        pairs = [(trial, rollouts.get(trial.trial)) for trial in trials]
        earlier = tuple(f for result in workpiece.evidence if result.review for f in earlier_findings(result.review))
        sources = evidence(submission, pairs, policy.setting("tools.output_chars", int))
        final = ("solve_review.final", partial(FinalReportTool, earlier=earlier), earlier_reports(workpiece))
        probe = ValidationResult(submission.digest, trials, round_.controls, ())
        review = None  # as in validation, a record review would not decide on is not reviewed (``reviewed``)
        if not isinstance(assess(probe, config.band, len(trials)), Reject | Incomplete):
            review = await solve_review(
                ctx, config, resources, workpiece.proposal, submission, pairs, *final, Billing.FINAL_REVIEW, sources
            )
    noted = (
        *(f for d in workpiece.decisions if isinstance(d, Accept | Revise) for f in d.notes),
        *(() if review is None else (*review.notes, *review.findings)),
    )
    notes = shipped_notes(sources, noted)
    result = ValidationResult(submission.digest, trials, round_.controls, (), review=review, confirms=index)
    return Validated(replace(result, notes=notes))
