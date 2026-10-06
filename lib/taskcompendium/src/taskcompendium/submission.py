# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator
from verifyit.json_objects import unique_object

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.grading_contract import (
    ActionSubmission,
    GradingAttempt,
    JsonSubmission,
    Submission,
    SubmissionFailure,
    TextSubmission,
    accepted_submission_types,
    resolve_verifier,
)
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    ConversationInput,
    TaskSpec,
    TextMessage,
    format_conversation,
)

ANSWER_CALL_NAME = "submit_answer"
ANSWER_FIELD = "answer"
JSON_VALUE = TypeAdapter(JsonValue, config=ConfigDict(strict=True, allow_inf_nan=False))


def answer_call_tool() -> dict[str, object]:
    """Return the function definition advertised by the answer-call convention."""
    return {
        "type": "function",
        "function": {
            "name": ANSWER_CALL_NAME,
            "description": "Submit the final answer to the task.",
            "parameters": {
                "type": "object",
                "properties": {ANSWER_FIELD: {"type": "string"}},
                "required": [ANSWER_FIELD],
                "additionalProperties": False,
            },
        },
    }


class SubmissionConvention(BaseModel, ABC):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    submission_types: ClassVar[tuple[type[Submission], ...]]

    @model_validator(mode="after")
    def validate_convention(self) -> Self:
        if not self.id:
            raise ValueError("A submission convention id is required")
        return self

    def supports(self, answer_type: AnswerType) -> bool:
        """Whether this convention can carry the semantic result."""
        return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)

    @abstractmethod
    def extract(self, attempt: GradingAttempt) -> Submission:
        """Read the agent's submission without access to expected values."""


class PlainText(SubmissionConvention):
    submission_types = (TextSubmission,)

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        return TextSubmission(_text_answer(attempt.conversation.events[-1]))


class JsonAnswer(SubmissionConvention):
    submission_types = (TextSubmission,)

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        try:
            value = _json_submission(_text_answer(attempt.conversation.events[-1]))
        except ValueError as error:
            raise SubmissionFailure("JSON submission is malformed") from error
        answer = value.get(ANSWER_FIELD) if isinstance(value, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise SubmissionFailure("JSON submission requires a nonempty string answer")
        return TextSubmission(answer)


class JsonValueAnswer(SubmissionConvention):
    """Parse the complete final assistant text as one JSON value."""

    submission_types = (JsonSubmission,)

    def supports(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.JSON

    def extract(self, attempt: GradingAttempt) -> JsonSubmission:
        try:
            return JsonSubmission(_json_submission(_text_answer(attempt.conversation.events[-1])))
        except ValueError as error:
            raise SubmissionFailure("JSON value submission is malformed") from error


class AnswerCall(SubmissionConvention):
    submission_types = (TextSubmission,)

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        response = attempt.conversation.events[-1]
        if (
            not isinstance(response, AssistantToolCalls)
            or len(response.calls) != 1
            or response.calls[0].name != ANSWER_CALL_NAME
        ):
            raise SubmissionFailure(f"Answer call requires one {ANSWER_CALL_NAME} function call")
        arguments = response.calls[0].arguments
        if (
            set(arguments) != {ANSWER_FIELD}
            or not isinstance(arguments[ANSWER_FIELD], str)
            or not arguments[ANSWER_FIELD].strip()
        ):
            raise SubmissionFailure("Answer call requires a nonempty string answer")
        return TextSubmission(arguments[ANSWER_FIELD])


class FinalAction(SubmissionConvention):
    submission_types = (ActionSubmission,)

    require_call: bool = False
    max_calls: int | None = Field(default=None, gt=0)

    def supports(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.NATIVE_ACTION

    def validate_final_message(self, response: ConversationEvent) -> TextMessage | AssistantToolCalls:
        """Require the assistant's final message to honor the call contract."""
        if not isinstance(response, (TextMessage, AssistantToolCalls)) or (
            isinstance(response, TextMessage) and response.role != "assistant"
        ):
            raise SubmissionFailure("Final action requires an assistant message")
        if self.require_call and not isinstance(response, AssistantToolCalls):
            raise SubmissionFailure("Final action requires a function call")
        if (
            isinstance(response, AssistantToolCalls)
            and self.max_calls is not None
            and len(response.calls) > self.max_calls
        ):
            raise SubmissionFailure(f"Final action permits at most {self.max_calls} function calls")
        return response

    def extract(self, attempt: GradingAttempt) -> ActionSubmission:
        return ActionSubmission(self.validate_final_message(attempt.conversation.events[-1]))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _json_submission(text: str) -> JsonValue:
    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=_reject_json_constant)
    return JSON_VALUE.validate_python(value)


def _text_answer(response: ConversationEvent) -> str:
    if not isinstance(response, TextMessage) or response.role != "assistant" or not response.content.strip():
        raise SubmissionFailure("Text submission requires nonempty assistant content without tool calls")
    return response.content


@dataclass(frozen=True)
class SubmissionCompatibility:
    """Whether a convention preserves the task's result contract, with reasons when it does not."""

    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.reasons


def submission_compatibility(specification: TaskSpec, convention: SubmissionConvention) -> SubmissionCompatibility:
    """Explain which parts of the task a submission convention cannot carry."""
    if not convention.supports(specification.answer_type):
        return SubmissionCompatibility((f"{type(convention).__name__} cannot carry {specification.answer_type.value}",))
    accepted = accepted_submission_types(resolve_verifier(specification.verifier))
    if not any(produced in accepted for produced in convention.submission_types):
        return SubmissionCompatibility(("Submission envelope is not accepted by the selected verifier",))
    if isinstance(convention, FinalAction):
        reasons = []
        if not specification.final_tools:
            reasons.append("final action requires at least one final tool")
        return SubmissionCompatibility(tuple(reasons))
    if isinstance(convention, AnswerCall):
        reasons = []
        if any(function.name == ANSWER_CALL_NAME for function in specification.final_tools):
            reasons.append("final tool name collides with submit_answer")
        return SubmissionCompatibility(tuple(reasons))
    return SubmissionCompatibility(())


def submission_instruction(convention: SubmissionConvention) -> str:
    """Return the instruction added after a conversation prefix."""
    match convention:
        case PlainText():
            return "Give your answer as plain text."
        case JsonAnswer():
            return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
        case JsonValueAnswer():
            return "Give your final answer as one JSON value, without Markdown fences."
        case AnswerCall():
            return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
        case FinalAction():
            return ""
        case _:
            raise ValueError(f"Unsupported submission convention: {type(convention).__name__}")


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return readable instruction text for the selected convention."""
    context = specification.context
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention {convention.id!r} is incompatible: {'; '.join(compatibility.reasons)}")
    if isinstance(convention, FinalAction):
        return format_conversation(context.events)
    return f"{format_conversation(context.events)}\n\n{submission_instruction(convention)}\n"


def conversation_messages(context: ConversationInput) -> list[dict[str, Any]]:
    """Convert the model-visible prefix to OpenAI-compatible chat messages."""
    messages: list[dict[str, Any]] = []
    for event in context.events:
        if isinstance(event, TextMessage):
            messages.append({"role": event.role, "content": event.content})
        elif isinstance(event, AssistantToolCalls):
            messages.append(
                {
                    "role": "assistant",
                    "content": event.content,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, separators=(",", ":"), ensure_ascii=False),
                            },
                        }
                        for call in event.calls
                    ],
                }
            )
        else:
            messages.append({"role": "tool", "tool_call_id": event.call_id, "content": event.content})
    return messages


def chat_request(specification: TaskSpec, convention: SubmissionConvention) -> dict[str, Any]:
    """Prepare the conversation and tools for the selected submission convention."""
    unsupported = unsupported_direct_chat_features(specification)
    if unsupported:
        raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention is incompatible: {'; '.join(compatibility.reasons)}")
    messages = conversation_messages(specification.context)
    instruction = submission_instruction(convention)
    if instruction:
        messages.append({"role": "user", "content": instruction})
    request: dict[str, Any] = {"messages": messages}
    tools: list[dict[str, object]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in specification.final_tools
    ]
    if isinstance(convention, AnswerCall):
        tools.append(answer_call_tool())
        request.update(tool_choice="required", parallel_tool_calls=False)
    if tools:
        request["tools"] = tools
    if isinstance(convention, FinalAction):
        if convention.require_call:
            request["tool_choice"] = "required"
        if convention.max_calls == 1:
            request["parallel_tool_calls"] = False
    return request
