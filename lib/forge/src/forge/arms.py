# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate-session and validate-tool arms, with pure budgeted transitions."""

from functools import partial
from types import MappingProxyType

from forge.config import Edge, Transitions
from forge.harness.protocol import Returned, Stopped, StopReason, Submitted
from forge.harness.tools.validate import ValidateRequest
from forge.model.llm import Message, ToolCall
from forge.model.review import Accept, Budget, Incomplete, Modify, Reject, RejectKind, Retry
from forge.model.serialization import VALIDATION_RESULT
from forge.model.stage import Exit, Stage
from forge.model.workpiece import Conversation, Terminal, Workpiece
from forge.policy import Policy
from forge.stages.exits import Decided, Validated


def submitted(wp: Workpiece, result: Exit) -> Workpiece:
    if not isinstance(result, Submitted):
        raise ValueError("Submission edge requires Submitted")
    return wp.with_(submission=result.submission, session=result.session)


def returned(wp: Workpiece, result: Exit) -> Workpiece:
    if not isinstance(result, Returned) or not isinstance(result.payload, ValidateRequest):
        raise ValueError("Validate edge requires Returned[ValidateRequest]")
    return wp.with_(submission=result.payload.submission, session=result.session)


def pending_validate(session: Conversation) -> ToolCall | None:
    answered = {message.tool_call_id for message in session.messages if message.role == "tool"}
    calls = (call for message in reversed(session.messages) for call in message.tool_calls)
    return next((call for call in calls if call.name == "validate" and call.id not in answered), None)


def validated(wp: Workpiece, result: Exit, *, tool: bool) -> Workpiece:
    if not isinstance(result, Validated):
        raise ValueError("Validation edge requires Validated")
    session = wp.session
    if tool and session is not None and (call := pending_validate(session)) is not None:
        message = Message("tool", VALIDATION_RESULT.dump_json(result.result).decode(), tool_call_id=call.id)
        session = Conversation((*session.messages, message), session.tool_schemas)
    return wp.with_(evidence=(*wp.evidence, result.result), session=session)


def after_validation(wp: Workpiece) -> Stage:
    return Stage.AUTHOR if wp.session and wp.session.messages[-1].role == "tool" else Stage.REVIEW


def decided(wp: Workpiece, result: Exit, *, budget: Budget, policy: Policy) -> Workpiece:
    if not isinstance(result, Decided):
        raise ValueError("Review edge requires Decided")
    decision = result.decision
    wp = wp.with_(decisions=(*wp.decisions, decision))
    match decision:
        case Accept():
            return wp.with_(terminal=Terminal.ACCEPTED)
        case Reject():
            return wp.with_(terminal=Terminal.REJECTED)
        case Modify(findings):
            if wp.round >= budget.max_modify_rounds:
                return wp.with_(terminal=Terminal.ABANDONED)
            previous = wp.session
            if previous is None:
                raise ValueError("Cannot revise without an author session")
            system = tuple(message for message in previous.messages if message.role == "system")
            prompt = policy.prompt("author.revision").render(kind="review", detail="\n".join(findings))
            return wp.with_(
                round=wp.round + 1, session=Conversation((*system, Message("user", prompt)), previous.tool_schemas)
            )
        case Retry():
            retries = sum(isinstance(item, Retry) for item in wp.decisions)
            return wp.with_(terminal=Terminal.HELD if retries > budget.max_validation_retries else None)
        case Incomplete():
            return wp.with_(terminal=Terminal.HELD)


def after_review(wp: Workpiece) -> Stage | None:
    if wp.terminal is not None:
        return None
    return Stage.VALIDATE if isinstance(wp.decisions[-1], Retry) else Stage.AUTHOR


def stopped(wp: Workpiece, result: Exit) -> Workpiece:
    if not isinstance(result, Stopped):
        raise ValueError("Stop edge requires Stopped")
    kind = RejectKind.BUDGET if result.reason == StopReason.CONTEXT_LIMIT else RejectKind.TASK
    decision = Reject(kind, (f"author stopped: {result.reason.value}",))
    return wp.with_(session=result.session, terminal=Terminal.ABANDONED, decisions=(*wp.decisions, decision))


def transitions(arm: str, budget: Budget, policy: Policy) -> Transitions:
    if arm not in ("S", "T"):
        raise ValueError(f"Unknown arm {arm!r}")
    edges: dict[tuple[Stage, type[Exit]], Edge] = {
        (Stage.AUTHOR, Submitted): (Stage.VALIDATE, submitted),
        (Stage.AUTHOR, Stopped): (Stage.AUTHOR, stopped),
        (Stage.VALIDATE, Validated): (
            after_validation if arm == "T" else Stage.REVIEW,
            partial(validated, tool=arm == "T"),
        ),
        (Stage.REVIEW, Decided): (after_review, partial(decided, budget=budget, policy=policy)),
    }
    if arm == "T":
        edges[(Stage.AUTHOR, Returned)] = (Stage.VALIDATE, returned)
    return MappingProxyType(edges)
