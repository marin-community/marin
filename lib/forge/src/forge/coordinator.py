# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one workpiece through its configured transition table; resume by replay."""

import logging

from forge import wiring
from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.effects.events import StageCompleted
from forge.effects.issues import Hold, Restart
from forge.effects.log import EventLog
from forge.model.content_hash import digest
from forge.model.proposal import TaskProposal
from forge.model.review import Incomplete
from forge.model.stage import Exit, Stage
from forge.model.workpiece import Terminal, Workpiece
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
        details.append(str(cause))
        cause = cause.__cause__
    return (" <- ".join(details),)


async def run_workpiece(proposal: TaskProposal, config: ForgeConfig, resources: Resources, log: EventLog) -> Workpiece:
    ctx = Context(log=log)
    wp = Workpiece(proposal.header.id, proposal)
    stage = config.initial_stage
    config_digest = digest(config.policy)
    interrupted = False
    stage_attempts: dict[tuple[Stage, int], int] = {}
    try:
        async with ctx.scope("workpiece"):
            wiring.install(ctx, resources)
            wp = wp.with_(id=await ctx.emit(NewId("workpiece")))
            while wp.terminal is None:
                key = (stage, wp.round)
                attempt = stage_attempts.get(key, 1)
                replay = False
                result: Exit | None = None
                try:
                    result = await STAGES[stage](ctx, wp, config, resources)
                except* Restart as error:
                    logging.warning("Stage %s round %d attempt %d of 3: %s", stage, wp.round, attempt, reasons(error))
                    if attempt >= 3:
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
                wp = adapt(wp, result)
                refs = {"round": str(wp.round), "terminal": wp.terminal or ""}
                await ctx.emit(StageCompleted(stage, config_digest, type(result).__name__, refs))
                following = target(wp) if callable(target) else target
                if following is None and wp.terminal is None:
                    raise ValueError("Transition stopped without a terminal workpiece")
                if following is not None:
                    stage = following
    except* Hold as error:
        interrupted = True
        wp = wp.with_(terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error))))
    except* Restart as error:
        interrupted = True
        wp = wp.with_(
            terminal=Terminal.HELD,
            decisions=(*wp.decisions, Incomplete(tuple(f"restart: {reason}" for reason in reasons(error)))),
        )
    finally:
        await ctx.drain()
    if interrupted:
        await log.record(StageCompleted(stage, config_digest, "Held", {"round": str(wp.round), "terminal": "held"}))
    return wp
