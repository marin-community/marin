# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate-session and validate-tool arms, with pure budgeted transitions."""

from functools import partial
from types import MappingProxyType

from forge.config import Transitions, transition
from forge.harness.protocol import Returned, Stopped, Submitted
from forge.harness.tools.validate import ValidateRequest
from forge.model.llm import Message, Role, ToolCall
from forge.model.review import Abandon, Accept, BandOutcome, Budget, Incomplete, Modify, Reject, Retry
from forge.model.stage import Stage
from forge.model.workpiece import Conversation, Terminal, Workpiece
from forge.policy import Policy
from forge.stages.exits import Decided, Validated
from forge.stages.summary import validation_path, validation_summary


def submitted(wp: Workpiece, result: Submitted) -> Workpiece:
    return wp.with_(submission=result.submission, session=result.session)


def returned(wp: Workpiece, result: Returned[ValidateRequest]) -> Workpiece:
    return wp.with_(submission=result.payload.submission, session=result.session)


def pending_validate(session: Conversation) -> ToolCall | None:
    answered = {message.tool_call_id for message in session.messages if message.role == "tool"}
    calls = (call for message in reversed(session.messages) for call in message.tool_calls)
    return next((call for call in calls if call.name == "validate" and call.id not in answered), None)


def validated(wp: Workpiece, result: Validated, *, tool: bool, workdir: str, output_chars: int) -> Workpiece:
    session = wp.session
    if tool and session is not None and (call := pending_validate(session)) is not None:
        content = validation_summary(result.result, validation_path(workdir, len(wp.evidence)), output_chars)
        message = Message(Role.TOOL, content, tool_call_id=call.id)
        session = Conversation((*session.messages, message), session.tool_schemas)
    return wp.with_(evidence=(*wp.evidence, result.result), session=session)


def after_validation(wp: Workpiece) -> Stage:
    return Stage.AUTHOR if wp.session and wp.session.messages[-1].role == "tool" else Stage.REVIEW


def decided(wp: Workpiece, result: Decided, *, budget: Budget, policy: Policy) -> Workpiece:
    decision = result.decision
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
            previous = wp.session
            if previous is None:
                raise ValueError("Cannot revise without an author session")
            system = tuple(message for message in previous.messages if message.role == "system")
            prompt = policy.prompt("author.revision").render(kind="review", detail="\n".join(findings))
            return wp.with_(
                round=wp.round + 1,
                session=Conversation((*system, Message(Role.USER, prompt)), previous.tool_schemas),
                too_easy_baseline=(
                    wp.submission if decision.placement is BandOutcome.TOO_EASY else wp.too_easy_baseline
                ),
            )
        case Retry():
            return wp
        case Incomplete():
            return wp.with_(terminal=Terminal.HELD)


def after_review(wp: Workpiece) -> Stage | None:
    if wp.terminal is not None:
        return None
    return Stage.VALIDATE if isinstance(wp.decisions[-1], Retry) else Stage.AUTHOR


def stopped(wp: Workpiece, result: Stopped) -> Workpiece:
    decision = Abandon((f"author stopped: {result.summary}",))
    return wp.with_(session=result.session, terminal=Terminal.ABANDONED, decisions=(*wp.decisions, decision))


def transitions(arm: str, budget: Budget, policy: Policy) -> Transitions:
    if arm not in ("S", "T"):
        raise ValueError(f"Unknown arm {arm!r}")
    edges = dict(
        (
            transition(Stage.AUTHOR, Submitted, Stage.VALIDATE, submitted),
            transition(Stage.AUTHOR, Stopped, None, stopped),
            transition(
                Stage.VALIDATE,
                Validated,
                after_validation if arm == "T" else Stage.REVIEW,
                partial(
                    validated,
                    tool=arm == "T",
                    workdir=policy.setting("author.workdir", str),
                    output_chars=policy.setting("tools.output_chars", int),
                ),
            ),
            transition(Stage.REVIEW, Decided, after_review, partial(decided, budget=budget, policy=policy)),
        )
    )
    if arm == "T":
        key, edge = transition(Stage.AUTHOR, Returned, Stage.VALIDATE, returned)
        edges[key] = edge
    return MappingProxyType(edges)
