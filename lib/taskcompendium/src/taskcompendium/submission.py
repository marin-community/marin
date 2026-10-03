# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission conventions for semantic answer tasks."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self

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
    ConversationToolCall,
    RawAssistantToolCalls,
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


class AnswerFormat(StrEnum):
    """The envelope used to deliver a result."""

    PLAIN = "plain"
    JSON = "json"
    JSON_VALUE = "json_value"
    ANSWER_CALL = "answer_call"
    FINAL_ACTION = "final_action"


class Convention(BaseModel, ABC):
    """How a result is requested, delivered, and extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    answer_format: AnswerFormat
    submission_types: ClassVar[tuple[type[Submission], ...]]

    @model_validator(mode="after")
    def validate_convention(self) -> Self:
        if not self.id:
            raise ValueError("A submission convention id is required")
        return self

    def supports(self, answer_type: AnswerType) -> bool:
        """Whether this convention can carry the semantic result."""
        if self.answer_format == AnswerFormat.JSON_VALUE:
            return answer_type == AnswerType.JSON
        if self.answer_format == AnswerFormat.FINAL_ACTION:
            return answer_type == AnswerType.NATIVE_ACTION
        return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)

    @abstractmethod
    async def extract(self, attempt: GradingAttempt) -> Submission:
        """Read the agent's submission without access to expected values."""


class PlainText(Convention):
    submission_types = (TextSubmission,)
    answer_format: Literal[AnswerFormat.PLAIN] = AnswerFormat.PLAIN

    async def extract(self, attempt: GradingAttempt) -> TextSubmission:
        return TextSubmission(_text_answer(attempt.conversation.events[-1]))


class JsonAnswer(Convention):
    submission_types = (TextSubmission,)
    answer_format: Literal[AnswerFormat.JSON] = AnswerFormat.JSON

    async def extract(self, attempt: GradingAttempt) -> TextSubmission:
        try:
            value = _json_submission(_text_answer(attempt.conversation.events[-1]))
        except ValueError as error:
            raise SubmissionFailure("JSON submission is malformed") from error
        answer = value.get(ANSWER_FIELD) if isinstance(value, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise SubmissionFailure("JSON submission requires a nonempty string answer")
        return TextSubmission(answer)


class JsonValueAnswer(Convention):
    """Parse the complete final assistant text as one JSON value."""

    submission_types = (JsonSubmission,)
    answer_format: Literal[AnswerFormat.JSON_VALUE] = AnswerFormat.JSON_VALUE

    async def extract(self, attempt: GradingAttempt) -> JsonSubmission:
        try:
            return JsonSubmission(_json_submission(_text_answer(attempt.conversation.events[-1])))
        except ValueError as error:
            raise SubmissionFailure("JSON value submission is malformed") from error


class AnswerCall(Convention):
    submission_types = (TextSubmission,)
    answer_format: Literal[AnswerFormat.ANSWER_CALL] = AnswerFormat.ANSWER_CALL

    async def extract(self, attempt: GradingAttempt) -> TextSubmission:
        response = _decoded_final_action(attempt.conversation.events[-1])
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


class FinalAction(Convention):
    submission_types = (ActionSubmission,)
    answer_format: Literal[AnswerFormat.FINAL_ACTION] = AnswerFormat.FINAL_ACTION

    require_call: bool = False
    max_calls: int | None = Field(default=None, gt=0)

    def validate_final_message(
        self, response: ConversationEvent | RawAssistantToolCalls
    ) -> TextMessage | AssistantToolCalls:
        """Require the assistant's final message to honor the call contract."""
        response = _decoded_final_action(response)
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

    async def extract(self, attempt: GradingAttempt) -> ActionSubmission:
        return ActionSubmission(self.validate_final_message(attempt.conversation.events[-1]))


SubmissionConvention = Annotated[
    PlainText | JsonAnswer | JsonValueAnswer | AnswerCall | FinalAction, Field(discriminator="answer_format")
]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _json_submission(text: str) -> JsonValue:
    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=_reject_json_constant)
    return JSON_VALUE.validate_python(value)


def _decoded_final_action(response: ConversationEvent | RawAssistantToolCalls) -> ConversationEvent:
    if not isinstance(response, RawAssistantToolCalls):
        return response
    try:
        calls = []
        for call in response.calls:
            arguments = _json_submission(call.arguments_json)
            if not isinstance(arguments, dict):
                raise ValueError("Function arguments must be a JSON object")
            calls.append(ConversationToolCall(call_id=call.call_id, name=call.name, arguments=arguments))
        return AssistantToolCalls(calls=tuple(calls), content=response.content)
    except ValueError as error:
        raise SubmissionFailure("Function arguments are malformed") from error


def _text_answer(response: ConversationEvent | RawAssistantToolCalls) -> str:
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


def submission_compatibility(specification: TaskSpec, convention: Convention) -> SubmissionCompatibility:
    """Explain which parts of the task a submission convention cannot carry."""
    if not convention.supports(specification.answer_type):
        return SubmissionCompatibility(
            (f"{convention.answer_format.value} cannot carry {specification.answer_type.value}",)
        )
    accepted = accepted_submission_types(resolve_verifier(specification.verifier))
    if not any(produced in accepted for produced in convention.submission_types):
        return SubmissionCompatibility(("Submission envelope is not accepted by the selected verifier",))
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        reasons = []
        if not specification.final_tools:
            reasons.append("final action requires at least one final tool")
        return SubmissionCompatibility(tuple(reasons))
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        reasons = []
        if any(function.name == ANSWER_CALL_NAME for function in specification.final_tools):
            reasons.append("final tool name collides with submit_answer")
        return SubmissionCompatibility(tuple(reasons))
    return SubmissionCompatibility(())


def submission_instruction(convention: SubmissionConvention) -> str:
    """Return the instruction added after a conversation prefix."""
    if convention.answer_format == AnswerFormat.PLAIN:
        return "Give your answer as plain text."
    if convention.answer_format == AnswerFormat.JSON:
        return f'Give your answer as a JSON object with an "{ANSWER_FIELD}" field.'
    if convention.answer_format == AnswerFormat.JSON_VALUE:
        return "Give your final answer as one JSON value, without Markdown fences."
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
        return f'Call {ANSWER_CALL_NAME} with your final answer as the "{ANSWER_FIELD}" string.'
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
        return ""
    raise ValueError(f"Unsupported answer format: {convention.answer_format}")


def render_instruction(specification: TaskSpec, convention: SubmissionConvention) -> str:
    """Return Harbor instruction text for the selected convention."""
    context = specification.context
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention {convention.id!r} is incompatible: {'; '.join(compatibility.reasons)}")
    if convention.answer_format == AnswerFormat.FINAL_ACTION:
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
    if convention.answer_format == AnswerFormat.ANSWER_CALL:
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
