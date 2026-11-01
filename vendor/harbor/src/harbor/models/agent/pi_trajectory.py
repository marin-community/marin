"""Validate the self-contained, in-process Pi execution and supervision record.

This is Pi's trial-local payload. Harbor adds the verifier result to the trial
result; consumers must never reconcile it against HTTP or teacher request logs.
"""

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PiModelEvidence(_Evidence):
    request_id: str
    model: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    chat_template_kwargs: dict[str, Any]
    assistant: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None

    @model_validator(mode="after")
    def complete(self) -> "PiModelEvidence":
        if (
            not self.request_id
            or not self.model
            or not self.messages
            or self.assistant.get("role") != "assistant"
            or not self.prompt_token_ids
            or not self.response_token_ids
            or (
                self.logprobs is not None
                and len(self.response_token_ids) != len(self.logprobs)
            )
            or any(
                type(token) is not int or token < 0
                for token in (*self.prompt_token_ids, *self.response_token_ids)
            )
            or (
                self.logprobs is not None
                and any(not math.isfinite(prob) for prob in self.logprobs)
            )
        ):
            raise ValueError(
                "Pi model evidence needs exact sampled IDs and an assistant response; present logprobs must align"
            )
        return self


class PiToolObservation(_Evidence):
    tool_call_id: str
    content: str
    authored_by: Literal["student", "teacher"]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Pi tool arguments contain duplicate JSON keys")
        result[key] = value
    return result


def same_pi_assistant(sampled: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """Check Pi's executed tool replay without confusing JSON whitespace with a changed action."""
    if sampled.get("role") != replayed.get("role") or sampled.get(
        "content"
    ) != replayed.get("content"):
        return False
    original, replay = sampled.get("tool_calls") or [], replayed.get("tool_calls") or []
    if (
        not isinstance(original, list)
        or not isinstance(replay, list)
        or len(original) != len(replay)
    ):
        return False
    for served, executed in zip(original, replay, strict=True):
        if not isinstance(served, dict) or not isinstance(executed, dict):
            return False
        if served.get("id") != executed.get("id") or served.get("type") != executed.get(
            "type"
        ):
            return False
        served_fn, executed_fn = served.get("function"), executed.get("function")
        if not isinstance(served_fn, dict) or not isinstance(executed_fn, dict):
            return False
        if served_fn.get("name") != executed_fn.get("name"):
            return False
        try:
            args = json.loads(
                served_fn["arguments"], object_pairs_hook=_unique_json_object
            )
            replay_args = json.loads(
                executed_fn["arguments"], object_pairs_hook=_unique_json_object
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(args, dict) or args != replay_args:
            return False
    return True


class PiExecutedTurn(_Evidence):
    index: int
    actor: Literal["student", "teacher", "revision"]
    executed: PiModelEvidence
    executed_assistant: dict[str, Any]
    observations: tuple[PiToolObservation, ...]
    teacher_label: PiModelEvidence | None

    @model_validator(mode="after")
    def attributed(self) -> "PiExecutedTurn":
        if self.actor != "teacher" and self.executed.logprobs is None:
            raise ValueError(
                "Pi student and revision behavior requires exact sampled logprobs"
            )
        if not same_pi_assistant(self.executed.assistant, self.executed_assistant):
            raise ValueError("Pi executed assistant differs from sampled action")
        calls = self.executed_assistant.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) != len(self.observations):
            raise ValueError("Pi turn requires every executed tool observation")
        for call, observation in zip(calls, self.observations, strict=True):
            if (
                not isinstance(call, dict)
                or call.get("id") != observation.tool_call_id
                or observation.authored_by
                != ("teacher" if self.actor == "teacher" else "student")
            ):
                raise ValueError(
                    "Pi tool observation does not match its executed author or tool call"
                )
        label = self.teacher_label
        if label is not None:
            if (
                label.messages != self.executed.messages
                or label.tools != self.executed.tools
                or label.chat_template_kwargs != self.executed.chat_template_kwargs
            ):
                raise ValueError(
                    "Pi private label and executed request saw different visited states"
                )
            if self.actor == "teacher" and label != self.executed:
                raise ValueError("Pi teacher execution must be the same sampled label")
            if self.actor != "teacher" and label.request_id == self.executed.request_id:
                raise ValueError("Pi student execution cannot reuse its private label")
        return self


class PiCollatedTrajectory(_Evidence):
    schema_version: Literal[1]
    trial_id: str
    policy_step: StrictInt
    turns: tuple[PiExecutedTurn, ...]
    planned_executors: tuple[Literal["student", "teacher"], ...] | None
    critique_feedback: str | None
    critique: dict[str, Any] | None

    @model_validator(mode="after")
    def continuous(self) -> "PiCollatedTrajectory":
        if (
            not self.trial_id
            or type(self.policy_step) is not int
            or self.policy_step < 0
            or not self.turns
        ):
            raise ValueError(
                "Pi trajectory needs trial identity, policy step and an executed model turn"
            )
        if self.planned_executors is not None and (
            len(self.turns) != len(self.planned_executors)
            or any(
                turn.actor != actor
                for turn, actor in zip(self.turns, self.planned_executors, strict=True)
            )
        ):
            raise ValueError("Pi executed actors do not match the pre-sampling plan")
        seen: set[str] = set()
        for index, turn in enumerate(self.turns):
            if turn.index != index or turn.executed.request_id in seen:
                raise ValueError(
                    "Pi trajectory needs ordered unique executed request IDs"
                )
            seen.add(turn.executed.request_id)
            if turn.teacher_label is not None and turn.actor != "teacher":
                if turn.teacher_label.request_id in seen:
                    raise ValueError("Pi private label needs a distinct request ID")
                seen.add(turn.teacher_label.request_id)
            if index:
                previous = self.turns[index - 1]
                context = (
                    *previous.executed.messages,
                    previous.executed_assistant,
                    *(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": result.content,
                        }
                        for result in previous.observations
                    ),
                )
                current = turn.executed.messages
                if current[: len(context)] != context:
                    raise ValueError(
                        "Pi continuation must include the exact executed action and observations"
                    )
                addition = current[len(context) :]
                permitted = (
                    ()
                    if self.critique_feedback is None
                    else (
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.critique_feedback}
                            ],
                        },
                    )
                )
                if addition != permitted:
                    raise ValueError(
                        "Pi continuation has missing or private unexecuted content"
                    )
        if self.critique is None:
            if self.critique_feedback is not None:
                raise ValueError(
                    "Pi private feedback needs an attributed teacher critique"
                )
        elif (
            len(self.turns) != 2
            or [turn.actor for turn in self.turns] != ["student", "revision"]
            or not self.critique_feedback
            or self.critique.get("feedback") != self.critique_feedback
            or self.critique.get("trial_id") != self.trial_id
            or self.critique.get("student_request_id")
            != self.turns[0].executed.request_id
            or not self.critique.get("teacher_request_id")
            or not self.critique.get("teacher_model")
            or self.turns[-1].observations
            or self.turns[-1].executed_assistant.get("tool_calls")
        ):
            raise ValueError(
                "Pi critique requires attributed teacher feedback and one tool-free student revision"
            )
        return self
