# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic task and its final submission."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.8"


class AnswerType(StrEnum):
    """The kind of result the task asks the model to produce."""

    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    STATE = "state"
    NATIVE_ACTION = "native_action"


class VerifierKind(StrEnum):
    """The registered grader used to check a submission."""

    EXACT_ANSWER = "exact_answer"
    PREDICTED_ACTION = "predicted_action"
    NUMERIC_ANSWER = "numeric_answer"
    MCQ_ANSWER = "mcq_answer"


class Source(BaseModel):
    """Pinned provenance for the source row and the importer that converted it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    revision: str
    row: str
    importer_revision: str

    @model_validator(mode="after")
    def validate_source(self) -> "Source":
        if not all((self.dataset, self.revision, self.row, self.importer_revision)):
            raise ValueError("Complete source provenance is required")
        return self


class VerifierSpec(BaseModel):
    """A private verifier selection and its pinned configuration.

    ``kind`` selects a verifier class. ``parameters_json`` is its private
    JSON-encoded configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: VerifierKind
    parameters_json: str = Field(repr=False)


class FunctionCall(BaseModel):
    """A protocol-independent function name and decoded argument object."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


@dataclass(frozen=True)
class ToolCallComparatorConfig:
    numeric_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.numeric_tolerance is not None and (
            isinstance(self.numeric_tolerance, bool)
            or not isfinite(self.numeric_tolerance)
            or self.numeric_tolerance < 0
        ):
            raise ValueError("Numeric tolerance must be finite and nonnegative")


class FunctionDefinition(BaseModel):
    """Function advertised to the model, without an execution binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    parameters: dict[str, JsonValue]
    description: str | None = None
    strict: bool | None = None


class TextMessage(BaseModel):
    """One source conversation turn sent to the model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["message"] = "message"
    role: str
    content: str

    @model_validator(mode="after")
    def validate_message(self) -> "TextMessage":
        if self.role not in {"system", "developer", "user", "assistant"}:
            raise ValueError("Conversation messages require a supported role")
        return self


class ConversationToolCall(BaseModel):
    """A function call with its conversation identity and decoded arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class AssistantToolCalls(BaseModel):
    """An assistant message containing function calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["assistant_tool_calls"] = "assistant_tool_calls"
    calls: tuple[ConversationToolCall, ...] = Field(min_length=1)
    content: str | None = None


class ToolResult(BaseModel):
    """A historical result for a function call in the conversation prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: str


type AssistantMessage = TextMessage | AssistantToolCalls


ConversationEvent = Annotated[TextMessage | AssistantToolCalls | ToolResult, Field(discriminator="type")]


def format_conversation(events: tuple[ConversationEvent, ...]) -> str:
    """Produce the Harbor instruction view of a structured conversation."""
    sections = []
    for event in events:
        if isinstance(event, TextMessage):
            sections.append(f"{event.role.title()}:\n{event.content.strip()}")
        elif isinstance(event, AssistantToolCalls):
            calls = "\n".join(f"{call.call_id}: {call.name}({call.arguments})" for call in event.calls)
            content = f"{event.content}\n" if event.content is not None else ""
            sections.append(f"Assistant:\n{content}{calls}")
        else:
            sections.append(f"Tool result {event.call_id}:\n{event.content}")
    return "\n\n".join(sections)


class ConversationInput(BaseModel):
    """Model-visible conversation prefix, without provider reasoning state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    events: tuple[ConversationEvent, ...]

    @model_validator(mode="after")
    def validate_input(self) -> "ConversationInput":
        if not self.events:
            raise ValueError("Conversation input requires events")
        pending: set[str] = set()
        seen: set[str] = set()
        for event in self.events:
            if isinstance(event, AssistantToolCalls):
                if pending or not event.calls:
                    raise ValueError("Historical calls require preceding results and a nonempty batch")
                for call in event.calls:
                    if not call.call_id or not call.name or call.call_id in seen:
                        raise ValueError("Historical call identifiers and names must be unique and nonempty")
                    pending.add(call.call_id)
                    seen.add(call.call_id)
            elif isinstance(event, ToolResult):
                if event.call_id not in pending:
                    raise ValueError("Historical tool result has no pending call")
                pending.remove(event.call_id)
            elif pending:
                raise ValueError("Historical calls require results before the next message")
            elif isinstance(event, TextMessage) and not event.content.strip():
                raise ValueError("Source conversation messages require nonempty content")
        if pending:
            raise ValueError("Historical calls require results before the final decision")
        return self


class ConversationTrace(BaseModel):
    """Complete model-visible conversation ending in an assistant submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    events: tuple[ConversationEvent, ...]

    @model_validator(mode="after")
    def validate_trace(self) -> "ConversationTrace":
        if len(self.events) < 2:
            raise ValueError("Grading evidence requires a prefix and final assistant message")
        ConversationInput(events=self.events[:-1])
        final = self.events[-1]
        if isinstance(final, ToolResult) or (isinstance(final, TextMessage) and final.role != "assistant"):
            raise ValueError("Grading evidence requires a final assistant message")
        if isinstance(final, AssistantToolCalls):
            identifiers = [call.call_id for call in final.calls]
            historical = {
                call.call_id
                for event in self.events[:-1]
                if isinstance(event, AssistantToolCalls)
                for call in event.calls
            }
            if len(set(identifiers)) != len(identifiers) or historical.intersection(identifiers):
                raise ValueError("Conversation call identifiers must be unique")
        return self


class TaskTools(BaseModel):
    """Functions and call policy advertised at the task's decision point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    functions: tuple[FunctionDefinition, ...] = ()
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None

    @model_validator(mode="after")
    def validate_tools(self) -> "TaskTools":
        if len({function.name for function in self.functions}) != len(self.functions):
            raise ValueError("Advertised function names must be unique")
        if self.tool_choice is not None and self.tool_choice not in {"auto", "none", "required"}:
            raise ValueError("Unsupported native tool choice")
        return self


class TaskRequirements(BaseModel):
    """Environment functionality required to run the task.

    ``capabilities`` contains generic operations such as ``filesystem`` or
    ``shell``. ``action_interfaces`` contains named stateful tool surfaces such
    as ``workplace:v1``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[str, ...] = ()
    action_interfaces: tuple[str, ...] = ()


class TaskSpec(BaseModel):
    """The private definition of one deterministic answer task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    context: ConversationInput
    requirements: TaskRequirements
    tools: TaskTools = Field(default_factory=TaskTools)
    answer_type: AnswerType
    verifier: VerifierSpec
    source: Source
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_specification(self) -> "TaskSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id:
            raise ValueError("A task id is required")
        if self.answer_type == AnswerType.NATIVE_ACTION and not self.tools.functions:
            raise ValueError("Native-action tasks require advertised functions")
        return self
