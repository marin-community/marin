# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private semantics for one deterministic task and its final submission."""

import base64
import binascii
import json
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from typing import Annotated, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from rigging.filesystem.path_validation import validate_relative_file_path, validate_relative_file_paths

from taskcompendium.environment import (
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    HealthcheckSpec,
)

SCHEMA_VERSION = "0.22"
DOCKER_IMAGE_PATTERN = r"^[^\s@]+@sha256:[0-9a-f]{64}$"
FILESYSTEM_CAPABILITY = "filesystem"
SHELL_CAPABILITY = "shell"


class AnswerType(StrEnum):
    """The kind of result the task asks the model to produce."""

    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    STATE = "state"
    WORKSPACE_STATE = "workspace_state"
    NATIVE_ACTION = "native_action"


class VerifierKind(StrEnum):
    """The grader kind used to check a submission."""

    EXACT_ANSWER = "exact"
    PREDICTED_ACTION = "predicted_action"
    NUMERIC_ANSWER = "numeric"
    MCQ_ANSWER = "mcq"
    SHELL = "shell"
    EXTERNAL = "external"
    STAGED = "staged"
    SKIPPED = "skipped"


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


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Verifier configuration contains a non-JSON numeric constant: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("Verifier configuration numbers must be finite")
    return result


class FunctionCall(BaseModel):
    """A protocol-independent function name and decoded argument object."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


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


class InlineFile(BaseModel):
    """File bytes encoded as canonical base64, including UTF-8 text files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["inline_file"] = "inline_file"
    content_base64: str

    @field_validator("content_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            payload = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid base64 resource content") from error
        if base64.b64encode(payload).decode("ascii") != value:
            raise ValueError("Base64 resource content must be canonical")
        return value


class TaskResource(BaseModel):
    """One inline file copied into a role's workspace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    source: InlineFile
    mode: str | None = Field(default=None, pattern=r"^[0-7]{3,4}$")
    mtime_ns: int | None = Field(default=None, strict=True)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        validate_relative_file_path(value)
        return value


class ResourceGroups(BaseModel):
    """Shared inputs and role-specific mounts, with independent private roots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    all: tuple[TaskResource, ...] = ()
    worker: tuple[TaskResource, ...] = ()
    oracle: tuple[TaskResource, ...] = ()
    verifier: tuple[TaskResource, ...] = ()

    @model_validator(mode="after")
    def validate_destinations(self) -> "ResourceGroups":
        for resources in (self.worker, self.oracle, self.verifier):
            validate_relative_file_paths(resource.path for resource in self.all + resources)
        return self


class ProviderRequirement(BaseModel):
    """One versioned action interface and literal JSON initial state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_interface: str = Field(min_length=1)
    initial_state: JsonValue = Field(repr=False)

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(cls, value: JsonValue) -> JsonValue:
        json.dumps(value, allow_nan=False)
        return value


def validate_workspace_path(path: str) -> PurePosixPath:
    """Require an absolute POSIX workspace path interpreted by the runtime."""
    workspace = PurePosixPath(path)
    if not workspace.is_absolute():
        raise ValueError(f"Workspace path must be absolute: {path!r}")
    return workspace


class EnvironmentRequirements(BaseModel):
    """Operations, pinned initial workspace, and named tool-provider contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[str, ...] = ()
    docker_image: str | None = Field(default=None, pattern=DOCKER_IMAGE_PATTERN)
    working_directory: str | None = None
    setup_commands: tuple[str, ...] = ()
    environment_variables: dict[str, str] = Field(default_factory=dict)
    tool_providers: dict[str, ProviderRequirement] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_environment(self) -> "EnvironmentRequirements":
        if any(not capability for capability in self.capabilities):
            raise ValueError("Capabilities must be nonempty names")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Capabilities must be unique")
        if any(not name for name in self.tool_providers):
            raise ValueError("Provider requirement names must be nonempty")
        if any(not command.strip() for command in self.setup_commands):
            raise ValueError("Setup commands must be nonempty")
        if self.working_directory is not None:
            validate_workspace_path(self.working_directory)
        return self


class VerifierSpec(BaseModel):
    """A private verifier selection and its pinned configuration.

    ``parameters_json`` belongs to the scorer. Typed environment requirements
    declare private harness capabilities independently of scoring configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1)
    parameters_json: str = Field(repr=False)
    environment_requirements: EnvironmentRequirements = Field(default_factory=EnvironmentRequirements)

    @field_validator("parameters_json")
    @classmethod
    def validate_parameters(cls, value: str) -> str:
        parameters = json.loads(value, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
        if not isinstance(parameters, dict):
            raise ValueError("Verifier configuration must be a JSON object")
        return value


class TaskStage(BaseModel):
    """One stage in a shared task machine. The first stage uses the task context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    context: ConversationInput | None = None
    verifier: VerifierSpec
    workdir_files: tuple[EnvironmentFile, ...] = ()
    setup: tuple[EnvironmentCommand, ...] = ()
    healthcheck: HealthcheckSpec | None = None
    agent_timeout: float | None = Field(default=None, gt=0)
    agent_user: str | None = None
    minimum_rewards: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(default_factory=dict)


class StageRewardStrategy(StrEnum):
    MEAN = "mean"
    FINAL = "final"


class StageVerifierSpec(BaseModel):
    """Reduce the grades from completed task stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: StageRewardStrategy


class SkippedVerifierSpec(BaseModel):
    """Explain why a runtime intentionally omitted grading."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(min_length=1)


class TaskSpec(BaseModel):
    """The private definition of one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    context: ConversationInput
    environment_requirements: EnvironmentRequirements
    final_tools: tuple[FunctionDefinition, ...] = ()
    interaction_tools: tuple[FunctionDefinition, ...] = ()
    output_paths: tuple[str, ...] = ()
    answer_type: AnswerType
    verifier: VerifierSpec
    environment: EnvironmentSpec = Field(default_factory=lambda: EnvironmentSpec(kind=EnvironmentKind.NULL))
    attempt_timeout: float | None = Field(default=None, gt=0)
    agent_timeout: float | None = Field(default=None, gt=0)
    agent_user: str | None = None
    stages: tuple[TaskStage, ...] = ()
    source: Source
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    resources: ResourceGroups = Field(default_factory=ResourceGroups)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_specification(self) -> "TaskSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported TaskSpec schema: {self.schema_version}")
        if not self.id:
            raise ValueError("A task id is required")
        if len({function.name for function in self.final_tools}) != len(self.final_tools):
            raise ValueError("Advertised function names must be unique")
        if self.answer_type == AnswerType.NATIVE_ACTION and not self.final_tools:
            raise ValueError("Native-action tasks require advertised functions")
        if (self.verifier.kind == VerifierKind.STAGED) != bool(self.stages):
            raise ValueError("Task stages require a staged verifier")
        if self.stages:
            if self.environment.kind == EnvironmentKind.NULL or self.environment.interaction is not None:
                raise ValueError("Task stages require a shared shell machine")
            if self.stages[0].context is not None or any(stage.context is None for stage in self.stages[1:]):
                raise ValueError("Only the first stage uses the task context")
            if len({stage.name for stage in self.stages}) != len(self.stages):
                raise ValueError("Task stage names must be unique")
            if any(stage.verifier.kind in {VerifierKind.STAGED, VerifierKind.EXTERNAL} for stage in self.stages):
                raise ValueError("Task stages require an executable private verifier")
        return self
