# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one workpiece through author, validate and review until it is terminal; resume by replay.

The author submits, validation gathers evidence for that submission, and review decides. A ``Modify``
decision starts the next round with a fresh author session; ``Retry`` validates the same submission
again.
"""

import logging
from collections.abc import Callable

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note, StageCompleted
from taskforge.effects.issues import Hold, Restart, Transient, describe_exception
from taskforge.effects.log import EventLog
from taskforge.harness.protocol import Stopped, Submitted
from taskforge.model.content_hash import digest
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Abandon, Accept, BandOutcome, Budget, Decision, Incomplete, Modify, Reject, Retry
from taskforge.model.stage import Exit, Stage
from taskforge.model.workpiece import Conversation, Terminal, Workpiece
from taskforge.policy import Policy
from taskforge.resources.bundle import Resources
from taskforge.resources.ids import NewId
from taskforge.stages import STAGES
from taskforge.stages.exits import Decided, Validated


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


def apply_decision(wp: Workpiece, decision: Decision, budget: Budget, policy: Policy) -> Workpiece:
    """Record a review decision and apply it: terminal, a revision round, or unchanged for a retry.

    A ``Modify`` past ``budget.max_modify_rounds`` becomes ``Abandon``. An ``Accept`` with ``revert``
    falls back to the too-easy baseline, the submission validated before the hardening round.
    """
    if isinstance(decision, Modify) and wp.round >= budget.max_modify_rounds:
        decision = Abandon((f"budget.max_modify_rounds exhausted after {wp.round} modify rounds", *decision.findings))
    wp = wp.with_(decisions=(*wp.decisions, decision))
    match decision:
        case Accept():
            if decision.revert and wp.too_easy_baseline is None:
                raise ValueError("Cannot revert hardening without the too-easy baseline")
            return wp.with_(
                terminal=Terminal.ACCEPTED,
                submission=wp.too_easy_baseline if decision.revert else wp.submission,
                accepted_with=BandOutcome.TOO_EASY if decision.band is BandOutcome.TOO_EASY else None,
            )
        case Reject():
            return wp.with_(terminal=Terminal.REJECTED)
        case Abandon():
            return wp.with_(terminal=Terminal.ABANDONED)
        case Modify(findings):
            if wp.session is None:
                raise ValueError("Cannot revise without an author session")
            system = tuple(message for message in wp.session.messages if message.role == "system")
            prompt = policy.prompt("author.revision").render(kind="review", detail="\n".join(findings))
            return wp.with_(
                round=wp.round + 1,
                session=Conversation((*system, Message(Role.USER, prompt)), wp.session.tool_schemas),
                too_easy_baseline=wp.submission if decision.placement is BandOutcome.TOO_EASY else wp.too_easy_baseline,
            )
        case Retry():
            return wp
        case Incomplete():
            return wp.with_(terminal=Terminal.HELD)


def advance(wp: Workpiece, stage: Stage, result: Exit, config: TaskforgeConfig) -> tuple[Workpiece, Stage | None]:
    """Apply ``stage``'s exit to the workpiece and pick the next stage, or None once the workpiece is terminal.

    Raises:
        ValueError: ``result`` is not an exit ``stage`` may return.
    """
    match stage, result:
        case Stage.AUTHOR, Submitted(submission, session):
            return wp.with_(submission=submission, session=session), Stage.VALIDATE
        case Stage.AUTHOR, Stopped():
            decision = Abandon((f"author stopped: {result.summary}",))
            stopped = wp.with_(session=result.session, terminal=Terminal.ABANDONED, decisions=(*wp.decisions, decision))
            return stopped, None
        case Stage.VALIDATE, Validated(evidence):
            return wp.with_(evidence=(*wp.evidence, evidence)), Stage.REVIEW
        case Stage.REVIEW, Decided(decision):
            wp = apply_decision(wp, decision, config.budget, config.policy)
            if wp.terminal is not None:
                return wp, None
            return wp, Stage.VALIDATE if isinstance(decision, Retry) else Stage.AUTHOR
    raise ValueError(f"{stage} returned {type(result).__name__}, which has no transition")


async def run_workpiece(
    proposal: TaskProposal,
    config: TaskforgeConfig,
    resources: Resources,
    log: EventLog,
    *,
    install: Callable[[Context, Resources], None],
) -> Workpiece:
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
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error))))
    finally:
        try:
            await ctx.drain()
        finally:
            await resources.close_round()
    if interrupted:
        await log.record(StageCompleted(stage, config_digest, "Held", wp.round, Terminal.HELD))
    return wp
