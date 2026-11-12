# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one workpiece through its configured transition table; resume by replay."""

import logging
from collections.abc import Callable

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import StageCompleted
from forge.effects.issues import Hold, Restart, Transient, describe_exception
from forge.effects.log import EventLog
from forge.model.content_hash import digest
from forge.model.proposal import TaskProposal
from forge.model.review import Incomplete
from forge.model.stage import Exit, Stage
from forge.model.workpiece import Submission, Terminal, Workpiece
from forge.resources.bundle import Resources
from forge.resources.ids import NewId
from forge.stages import STAGES


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
"""Bound interrupted-stage reconstruction; effect executors alone own retries."""


async def run_workpiece(
    proposal: TaskProposal,
    config: ForgeConfig,
    resources: Resources,
    log: EventLog,
    *,
    install: Callable[[Context, Resources], None],
    seed: Submission | None = None,
) -> Workpiece:
    """Run ``proposal`` from the configured initial stage, or from validation of a ``seed`` submission."""
    ctx = Context(log=log)
    wp = Workpiece(proposal.header.id, proposal, submission=seed)
    stage = config.initial_stage if seed is None else Stage.VALIDATE
    config_digest = digest(config.policy)
    interrupted = False
    stage_attempts: dict[tuple[Stage, int], int] = {}
    try:
        async with ctx.scope("workpiece"):
            install(ctx, resources)
            wp = wp.with_(id=await ctx.emit(NewId("workpiece")))
            while wp.terminal is None:
                key = (stage, wp.round)
                attempt = stage_attempts.get(key, 1)
                replay = False
                result: Exit | None = None
                try:
                    result = await STAGES[stage](ctx, wp, config, resources)
                except* (Restart, Transient) as error:
                    logging.warning(
                        "Stage %s round %d attempt %d of %d: %s",
                        stage,
                        wp.round,
                        attempt,
                        STAGE_REPLAY_ATTEMPTS,
                        reasons(error),
                    )
                    if attempt >= STAGE_REPLAY_ATTEMPTS:
                        raise
                    stage_attempts[key] = attempt + 1
                    replay = True
                if replay:
                    continue
                assert result is not None
                edge = config.transitions.get((stage, type(result)))
                if edge is None:
                    raise ValueError(f"No transition for {stage}/{type(result).__name__}")
                target, adapt = edge
                previous_round = wp.round
                wp = adapt(wp, result)
                if wp.round != previous_round:
                    await resources.close_round()
                await ctx.emit(StageCompleted(stage, config_digest, type(result).__name__, wp.round, wp.terminal))
                following = target(wp) if callable(target) else target
                if following is None and wp.terminal is None:
                    raise ValueError("Transition stopped without a terminal workpiece")
                if following is not None:
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
