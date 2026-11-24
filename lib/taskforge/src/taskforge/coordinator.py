# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one workpiece through author, validate and review until it is terminal; resume by replay.

The author submits, validation gathers evidence for that submission, and review decides. A ``Modify``
decision starts the next round with a fresh author session; ``Retry`` validates the same submission
again. An ``Accept`` goes on to the confirmation step, whose record review decides once more: a final
acceptance, or the final review's one ``Revise``, which starts a round as ``Modify`` does.
"""

import logging
from collections.abc import Callable
from dataclasses import replace

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note, StageCompleted
from taskforge.effects.issues import DecisionModelUnavailable, Hold, Restart, Transient, describe_exception
from taskforge.effects.log import EventLog
from taskforge.harness.protocol import Stopped, Submitted
from taskforge.model.content_hash import digest
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    Abandon,
    Accept,
    BandOutcome,
    Budget,
    Decision,
    Decline,
    FindingResponse,
    Incomplete,
    Modify,
    Reject,
    Response,
    Retry,
    Revise,
)
from taskforge.model.stage import Exit, Stage
from taskforge.model.workpiece import Conversation, Fallback, Terminal, Workpiece
from taskforge.policy import Policy
from taskforge.resources.bundle import Resources
from taskforge.resources.ids import NewId
from taskforge.stages import STAGES
from taskforge.stages.exits import Decided, Validated
from taskforge.stages.rules import author_stopped
from taskforge.stages.solve_review import finding_lines


def reasons(error: BaseException) -> tuple[str, ...]:
    if isinstance(error, BaseExceptionGroup):
        return tuple(reason for child in error.exceptions for reason in reasons(child))
    details: list[str] = []
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        details.append(describe_exception(cause))
        cause = cause.__cause__
    return (" <- ".join(details),)


STAGE_REPLAY_ATTEMPTS = 3
"""Bound consecutive interrupted reconstructions of one stage visit; effect executors alone own retries."""


def digest_of(fallback: Fallback | None) -> str | None:
    return None if fallback is None else fallback.submission.digest


def apply_decision(wp: Workpiece, decision: Decision, budget: Budget, policy: Policy) -> Workpiece:
    """Record a review decision and apply it: terminal, a revision round, or unchanged for a retry.

    A ``Modify`` past ``budget.max_modify_rounds`` becomes ``Abandon``; one keeps the fallback its ``fallback`` digest
    names. An ``Accept`` with ``restore`` accepts the fallback, which must carry that digest; it is terminal once it
    decides on a confirmation record, or when ``solver.confirmation_k`` is 0.
    """
    if isinstance(decision, Modify) and wp.round >= budget.max_modify_rounds:
        decision = Abandon((f"budget.max_modify_rounds exhausted after {wp.round} modify rounds", *decision.findings))
    wp = wp.with_(decisions=(*wp.decisions, decision))
    match decision:
        case Accept(band, _, restore):
            if restore is not None and (
                wp.fallback is None or (digest_of(wp.fallback), wp.fallback.band) != (restore, band)
            ):
                raise ValueError(f"Cannot restore {restore} at {band}: the fallback is {wp.fallback}")
            if restore is not None and wp.fallback is not None:  # the revisions after the restored round are discarded
                after = wp.fallback.evidence
                wp = wp.with_(
                    responses=tuple(replace(r, discarded=r.discarded or r.review >= after) for r in wp.responses)
                )
            confirmed = bool(wp.evidence) and wp.evidence[-1].confirms is not None
            final = confirmed or not policy.setting("solver.confirmation_k", int)
            return wp.with_(  # a pending acceptance's placement is its decision's band, published only once final
                terminal=Terminal.ACCEPTED if final else None,
                submission=wp.fallback.submission if wp.fallback and restore else wp.submission,
                accepted_with=band if final and band is not BandOutcome.IN_BAND else None,
            )
        case Reject():
            return wp.with_(terminal=Terminal.REJECTED)
        case Abandon():
            return wp.with_(terminal=Terminal.ABANDONED)
        case Modify(findings, placement, _, _, review) | Revise(findings, review, placement=placement):
            if wp.session is None:
                raise ValueError("Cannot revise without an author session")
            fallback = decision.fallback if isinstance(decision, Modify) else None  # the final revision keeps none
            current = Fallback(wp.submission, placement, len(wp.evidence) - 1) if wp.submission and placement else None
            named = [option for option in (current, wp.fallback) if option and digest_of(option) == fallback]
            kept = named[0] if named else None  # the current submission over a stored one of its digest
            if digest_of(kept) != fallback:
                raise ValueError(
                    f"Cannot keep {fallback} as the fallback: it is neither the submission nor the fallback"
                )
            system = tuple(message for message in wp.session.messages if message.role == "system")
            notes = decision.notes if isinstance(decision, Revise) else ()
            detail = "\n".join(findings)
            if review:
                detail += "\n\n" + policy.prompt("author.review").render(findings=finding_lines(review))
            if notes:  # numbered after the revise findings, so ``declined`` names either kind
                detail += "\n\n" + policy.prompt("author.notes").render(notes=finding_lines(notes, len(review)))
            kind = "review" if isinstance(decision, Modify) else "the final review"
            prompt = policy.prompt("author.revision").render(kind=kind, detail=detail)
            sent = (FindingResponse(len(wp.evidence) - 1, f, Response.REQUESTED) for f in (*review, *notes))
            return wp.with_(
                round=wp.round + 1,
                session=Conversation((*system, Message(Role.USER, prompt)), wp.session.tool_schemas),
                fallback=kept,
                responses=(*wp.responses, *sent),
            )
        case Retry():
            return wp
        case Incomplete():
            return wp.with_(terminal=Terminal.HELD)


def answered(wp: Workpiece, declined: tuple[Decline, ...], response: Response) -> Workpiece:
    """``wp`` with each finding still requested of the author answered: declined with its reason, else ``response``."""
    reasons = {decline.finding: decline.reason for decline in declined}
    return wp.with_(
        responses=tuple(
            (
                r
                if r.response is not Response.REQUESTED
                else (
                    replace(r, response=Response.DECLINED, reason=reasons[r.finding])
                    if r.finding in reasons
                    else replace(r, response=response)
                )
            )
            for r in wp.responses
        )
    )


def advance(wp: Workpiece, stage: Stage, result: Exit, config: TaskforgeConfig) -> tuple[Workpiece, Stage | None]:
    """Apply ``stage``'s exit to the workpiece and pick the next stage, or None once the workpiece is terminal.

    Raises:
        ValueError: ``result`` is not an exit ``stage`` may return.
    """
    match stage, result:
        case Stage.AUTHOR, Submitted(submission, session):
            wp = answered(wp, submission.declined, Response.REVISED)
            return wp.with_(submission=submission, session=session), Stage.VALIDATE
        case Stage.AUTHOR, Stopped():
            wp = answered(wp, (), Response.STOPPED)
            stopped = apply_decision(
                wp.with_(session=result.session), author_stopped(wp, result.summary), config.budget, config.policy
            )
            return stopped, None if stopped.terminal is not None else Stage.CONFIRM
        case Stage.VALIDATE | Stage.CONFIRM, Validated(evidence):
            return wp.with_(evidence=(*wp.evidence, evidence)), Stage.REVIEW
        case Stage.REVIEW, Decided(decision):
            wp = apply_decision(wp, decision, config.budget, config.policy)
            if wp.terminal is not None:
                return wp, None
            following = Stage.CONFIRM if isinstance(decision, Accept) else Stage.AUTHOR
            return wp, Stage.VALIDATE if isinstance(decision, Retry) else following
    raise ValueError(f"{stage} returned {type(result).__name__}, which has no transition")


async def run_workpiece(
    proposal: TaskProposal,
    config: TaskforgeConfig,
    resources: Resources,
    log: EventLog,
    *,
    install: Callable[[Context, Resources], None],
    reached: Callable[[Workpiece], None] = lambda _: None,
) -> Workpiece:
    """``reached`` sees the workpiece each stage visit starts from, for a caller's diagnosis of an unexpected failure."""
    ctx = Context(log=log)
    wp = Workpiece(proposal.header.id, proposal)
    stage = Stage.AUTHOR
    config_digest = digest(config.policy)
    interrupted = False
    attempt = 1  # of the executing stage visit; a failed attempt replays the same stage and round
    try:
        async with ctx.scope("workpiece"):
            install(ctx, resources)
            wp = wp.with_(id=await ctx.emit(NewId("workpiece")))
            while True:
                reached(wp)
                try:
                    result = await STAGES[stage](ctx, wp, config, resources)
                except* (Restart, Transient) as error:
                    # Journal each attempt's cause; a hold reason names only the last one.
                    cause = f"Stage {stage} round {wp.round} attempt {attempt} of {STAGE_REPLAY_ATTEMPTS}: " + (
                        "; ".join(reasons(error))
                    )
                    logging.warning("%s", cause)
                    await ctx.emit(Note(cause))
                    if attempt >= STAGE_REPLAY_ATTEMPTS:
                        raise
                    attempt += 1  # replay the interrupted stage
                else:
                    # A completed visit resets the budget, so a same-round re-entry starts with a full one.
                    attempt = 1
                    previous_round = wp.round
                    wp, following = advance(wp, stage, result, config)
                    if wp.round != previous_round:
                        await resources.close_round()
                    await ctx.emit(StageCompleted(stage, config_digest, type(result).__name__, wp.round, wp.terminal))
                    if following is None:
                        break  # keep the terminal stage, which a hold raised while the scope exits records
                    stage = following
    except* (Hold, Restart, Transient) as error:
        interrupted = True
        # Only a hold every leaf of which is a decision-model outage resumes without a person.
        awaits = DecisionModelUnavailable.awaits if error.split(DecisionModelUnavailable)[1] is None else None
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error), awaits=awaits)))
    finally:
        try:
            await ctx.drain()
        finally:
            await resources.close_round()
    if interrupted:
        await log.record(StageCompleted(stage, config_digest, "Held", wp.round, Terminal.HELD))
    return wp
