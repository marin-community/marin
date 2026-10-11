# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One deterministic task: its conversation, environment, grader, and answer format.

A grader is one of four kinds. ``VerifyitGrader`` names a verifyit mode, graded in process
against the extracted answer or, with an environment, by the verifyit command in a fresh
machine. ``ScriptGrader`` runs a command in a fresh machine and reads its reward. A
``SessionGrader`` task is graded by the rollout session that runs it. ``NoGrader`` records why a
task cannot be graded. The answer format says how the final answer is requested from the model
and extracted from its conversation.
"""

import base64
import binascii
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, ClassVar, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator, model_validator
from rigging.filesystem.path_validation import validate_relative_file_path, validate_relative_file_paths
from shellbox.machine import UnsupportedMachineSpec
from verifyit.candidate import candidate_spec
from verifyit.grade import InvalidTask
from verifyit.json_objects import unique_object
from verifyit.modes.extract import extract_boxed
from verifyit.spec import (
    DEFAULT_OUTPUT,
    DEFAULT_WORKSPACE,
    GotestSpec,
    JunitSpec,
    PytestSpec,
    ScriptSpec,
    Spec,
    StdioSpec,
    spec_from_table,
)

SCHEMA_VERSION = "0.26"
DOCKER_IMAGE_PATTERN = r"^[^\s@]+@sha256:[0-9a-f]{64}$"


class AnswerType(StrEnum):
    """The kind of result the task asks the model to produce."""

    TEXT = "text"
    NUMBER = "number"
    JSON = "json"
    FILE = "file"
    STATE = "state"
    WORKSPACE_STATE = "workspace_state"
    NATIVE_ACTION = "native_action"


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
    raise ValueError(f"Non-JSON numeric constant: {value}")


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
    """Format a structured conversation as readable instruction text."""
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
    """One inline regular file in a workspace or build context."""

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


class DockerBuildContext(BaseModel):
    """Unresolved Docker build inputs, preserved as data without executing the recipe.

    Paths are relative to the context root, including ``Dockerfile``. Retaining the
    recipe does not pin mutable base images or downloads made during a build.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: tuple[TaskResource, ...]

    @model_validator(mode="after")
    def validate_files(self) -> "DockerBuildContext":
        validate_relative_file_paths(resource.path for resource in self.files)
        if not any(resource.path == "Dockerfile" for resource in self.files):
            raise ValueError("A Docker build context requires Dockerfile")
        return self


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


class CommandSemantics(StrEnum):
    """The execution behavior a task's commands require."""

    LINUX_PROCESS = "linux_process"
    """Real Linux processes, installed executables, and a native filesystem."""
    SHELL_SIMULATOR = "shell_simulator"
    """A built-in shell language and virtual filesystem, without native executables or guest networking."""


class EnvironmentRequirements(BaseModel):
    """Operations, initial workspace or unresolved recipe, and tool-provider contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: tuple[str, ...] = ()
    command_semantics: CommandSemantics | None = None
    docker_image: str | None = Field(default=None, pattern=DOCKER_IMAGE_PATTERN)
    docker_build: DockerBuildContext | None = None
    working_directory: str | None = None
    setup_commands: tuple[str, ...] = ()
    environment_variables: dict[str, str] = Field(default_factory=dict)
    tool_providers: dict[str, ProviderRequirement] = Field(default_factory=dict)
    packages_lock: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_environment(self) -> "EnvironmentRequirements":
        if sum(value is not None for value in (self.docker_image, self.docker_build, self.packages_lock)) > 1:
            raise ValueError("Docker image, build context, and packages lock are mutually exclusive")
        if any(value is not None for value in (self.docker_image, self.docker_build, self.packages_lock)):
            if self.command_semantics != CommandSemantics.LINUX_PROCESS:
                raise ValueError("Native environment dependencies require Linux process semantics")
        if self.command_semantics == CommandSemantics.SHELL_SIMULATOR and set(self.capabilities) - {
            "shell",
            "filesystem",
        }:
            raise ValueError("The shell simulator provides only shell and filesystem capabilities")
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


def require_resolved_environment(requirements: EnvironmentRequirements) -> None:
    """Reject an unbuilt recipe before choosing or creating an execution environment."""
    if requirements.docker_build is not None:
        raise UnsupportedMachineSpec("Unresolved Docker build context must be built and pinned before execution")


GRADER_ROOTS = ("/tests", "/logs/verifier")
"""Paths the grading machine reserves for grader files and verdicts."""


def normalized_absolute_path(value: str) -> str:
    """Require an absolute POSIX path without ``.``, ``..``, or repeated separators."""
    path = validate_workspace_path(value)
    if ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Path must be normalized: {value!r}")
    return value


def under_grader_root(path: str) -> bool:
    return any(PurePosixPath(path).is_relative_to(root) for root in GRADER_ROOTS)


def validate_output_paths(paths: tuple[str, ...]) -> None:
    """Require normalized output selections disjoint from private grading roots."""
    private_roots = (*GRADER_ROOTS, "/solution")
    for path in paths:
        root = PurePosixPath(normalized_absolute_path(path))
        if any(root.is_relative_to(private) or PurePosixPath(private).is_relative_to(root) for private in private_roots):
            raise ValueError(f"Output paths overlap private mounts: {path}")


def _absolute_file_path(value: str) -> str:
    if not PurePosixPath(value).is_absolute():
        raise ValueError(f"Grader paths must be absolute: {value!r}")
    validate_relative_file_path(value.removeprefix("/"))
    return value


class VerifierCommand(BaseModel):
    """A trusted command run as root on the agent's machine to collect grader inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class ArtifactKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    AUTO = "auto"


class MissingArtifactPolicy(StrEnum):
    ERROR = "error"
    SKIP = "skip"


class VerifierArtifact(BaseModel):
    """An agent file or directory copied into the grading machine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    kind: ArtifactKind
    exclude: tuple[str, ...] = ()
    missing: MissingArtifactPolicy = MissingArtifactPolicy.ERROR

    _validate_paths = field_validator("source", "target")(_absolute_file_path)


class StdoutReward(BaseModel):
    """The command exits zero and its last nonempty stdout line is one finite number.

    Earlier lines, such as library output, are ignored. The runtime retains only the first 16 KiB of
    stdout, so a grader keeps its output below that for its reward line to be read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["stdout"] = "stdout"


class ExitCodeReward(BaseModel):
    """Score a completed command as one on success and zero on failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["exit_code"] = "exit_code"


class RewardFileFormat(StrEnum):
    NUMBER = "number"
    JSON = "json"


class RewardFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    format: RewardFileFormat
    key: str = "reward"

    _validate_path = field_validator("path")(_absolute_file_path)


class FileReward(BaseModel):
    """The first existing file supplies the score; a missing, empty, or invalid file is a grading failure.

    A JSON file's ``detail`` object, when present, becomes the grade detail.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["file"] = "file"
    files: tuple[RewardFile, ...] = Field(min_length=1)
    pass_above: float | None = Field(default=None, allow_inf_nan=False)


def _require_grader_environment(environment: EnvironmentRequirements) -> None:
    if environment.docker_image is None and environment.packages_lock is None and environment.docker_build is None:
        raise ValueError(
            "A grading environment requires a digest-pinned image, packages lock, or unresolved build context"
        )
    if environment.tool_providers:
        raise ValueError("A grading environment cannot declare tool providers")


def _require_finite_json(value: JsonValue) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except ValueError as error:
        raise ValueError("Grader configuration numbers must be finite") from error


def verifyit_answer_file(spec: Spec) -> str | None:
    """The file a verifyit mode reads its answer from, or ``None`` for a workspace mode."""
    if isinstance(spec, StdioSpec | PytestSpec | JunitSpec | GotestSpec):
        return None
    if isinstance(spec, ScriptSpec):
        return DEFAULT_OUTPUT
    return spec.output


class VerifyitGrader(BaseModel):
    """A verifyit mode and its configuration table, without the ``mode`` key.

    Without an environment the mode must be one verifyit grades in process, against the answer
    extracted from the conversation. With an environment, the verifyit command grades in a fresh
    machine built from it, with verifier resources under ``/tests``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["verifyit"] = "verifyit"
    mode: str
    parameters: dict[str, JsonValue] = Field(repr=False)
    environment: EnvironmentRequirements | None = None

    @model_validator(mode="after")
    def validate_grader(self) -> "VerifyitGrader":
        spec = verifyit_spec(self)
        if self.environment is not None:
            _require_grader_environment(self.environment)
            answer = verifyit_answer_file(spec)
            if answer is not None and under_grader_root(answer):
                raise ValueError("The answer file must lie outside /tests and /logs/verifier")
        return self


def verifyit_spec(grader: VerifyitGrader) -> Spec:
    """Build the verifyit specification; in-process modes also pass their mode's validation."""
    try:
        _require_finite_json(grader.parameters)
        if grader.environment is None:
            return candidate_spec(grader.mode, grader.parameters)
        if "mode" in grader.parameters:
            raise ValueError("Verifier parameters must not override the mode")
        return spec_from_table({"mode": grader.mode, **grader.parameters})
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {grader.mode!r} verifier parameters: {error}") from error


class ScriptGrader(BaseModel):
    """A command run in a fresh machine built from ``environment``; ``reward`` says how it scores.

    Before ``argv`` runs in ``cwd``, ``collect`` commands run on the agent's machine, ``artifacts``
    are copied from it, verifier resources are installed under ``/tests``, the extracted answer is
    written to ``answer_path`` (``None`` when the agent's files are the result), and the
    conversation is written to ``conversation_path`` as JSON chat messages.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["script"] = "script"
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = "/app"
    env: dict[str, str] = Field(default_factory=dict)
    environment: EnvironmentRequirements
    collect: tuple[VerifierCommand, ...] = ()
    artifacts: tuple[VerifierArtifact, ...] = ()
    answer_path: str | None = "/app/answer.txt"
    conversation_path: str = "/tests/conversation.json"
    reward: Annotated[StdoutReward | ExitCodeReward | FileReward, Field(discriminator="kind")] = StdoutReward()
    timeout: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def validate_grader(self) -> "ScriptGrader":
        _require_grader_environment(self.environment)
        normalized_absolute_path(self.cwd)
        normalized_absolute_path(self.conversation_path)
        if self.answer_path is not None:
            normalized_absolute_path(self.answer_path)
            if under_grader_root(self.answer_path):
                raise ValueError("The answer path must lie outside /tests and /logs/verifier")
            if self.answer_path == self.conversation_path:
                raise ValueError("The answer and conversation paths must differ")
        return self


class SessionGrader(BaseModel):
    """The rollout session registered for the task grades it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["session"] = "session"


class NoGrader(BaseModel):
    """A task without a runnable grader; ``contract`` keeps the source's grading terms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none"] = "none"
    reason: str = Field(min_length=1)
    contract: dict[str, JsonValue] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def validate_grader(self) -> "NoGrader":
        _require_finite_json(self.contract)
        return self


Grader = Annotated[VerifyitGrader | ScriptGrader | SessionGrader | NoGrader, Field(discriminator="kind")]


def grades_in_process(grader: Grader) -> bool:
    """Whether the grader scores the extracted answer without a grading machine."""
    return isinstance(grader, VerifyitGrader) and grader.environment is None


def grader_workspace(grader: Grader) -> str:
    """The directory the grader treats as the agent's workspace."""
    if isinstance(grader, ScriptGrader):
        return grader.cwd
    if isinstance(grader, VerifyitGrader):
        spec = verifyit_spec(grader)
        if isinstance(spec, StdioSpec | PytestSpec | JunitSpec | GotestSpec | ScriptSpec):
            return spec.workspace
    return DEFAULT_WORKSPACE


@dataclass(frozen=True)
class TextSubmission:
    value: str


@dataclass(frozen=True)
class ActionSubmission:
    message: TextMessage | AssistantToolCalls


@dataclass(frozen=True)
class JsonSubmission:
    value: JsonValue


@dataclass(frozen=True)
class StateSubmission:
    value: JsonValue


type Submission = TextSubmission | ActionSubmission | JsonSubmission | StateSubmission


class SubmissionFailure(ValueError):
    """The agent ended the interaction without a valid submission."""


@dataclass(frozen=True)
class GradingAttempt:
    """Captured trial evidence; state absence is distinct from captured JSON null."""

    conversation: ConversationTrace
    files: Mapping[str, bytes] = field(default_factory=dict)
    state: StateSubmission | None = None


JSON_VALUE = TypeAdapter(JsonValue, config=ConfigDict(strict=True, allow_inf_nan=False))


def decode_json_value(text: str) -> JsonValue:
    """Decode finite JSON evidence with unique object keys at every nesting level."""
    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=_reject_json_constant)
    return JSON_VALUE.validate_python(value)


CONVERSATION_ANSWERS = frozenset({AnswerType.TEXT, AnswerType.NUMBER, AnswerType.JSON, AnswerType.NATIVE_ACTION})
"""Answer types the final assistant message carries, as opposed to files or state."""


def _text_answer(response: ConversationEvent) -> str:
    if not isinstance(response, TextMessage) or response.role != "assistant" or not response.content.strip():
        raise SubmissionFailure("Text submission requires nonempty assistant content without tool calls")
    return response.content


class BaseAnswerFormat(BaseModel, ABC):
    """How the final answer is requested from the model and extracted from its conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_types: ClassVar[tuple[type[Submission], ...]]

    def supports(self, answer_type: AnswerType) -> bool:
        """Whether this format can carry the semantic result."""
        return answer_type in (AnswerType.TEXT, AnswerType.NUMBER)

    @abstractmethod
    def extract(self, attempt: GradingAttempt) -> Submission:
        """Read the agent's submission without access to expected values."""


class PlainText(BaseAnswerFormat):
    """The whole final assistant message."""

    kind: Literal["plain_text"] = "plain_text"
    submission_types = (TextSubmission,)

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        return TextSubmission(_text_answer(attempt.conversation.events[-1]))


class Boxed(BaseAnswerFormat):
    r"""The content of the last ``\boxed{...}`` in the final assistant message, else the whole message."""

    kind: Literal["boxed"] = "boxed"
    submission_types = (TextSubmission,)

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        text = _text_answer(attempt.conversation.events[-1])
        boxed = extract_boxed(text)
        return TextSubmission(text if boxed is None else boxed)


ANSWER_CALL_NAME = "submit_answer"
ANSWER_FIELD = "answer"


class JsonAnswer(BaseAnswerFormat):
    """The string ``answer`` field of a JSON object in the final assistant message."""

    kind: Literal["json_answer"] = "json_answer"
    submission_types = (TextSubmission,)

    def extract(self, attempt: GradingAttempt) -> TextSubmission:
        try:
            value = decode_json_value(_text_answer(attempt.conversation.events[-1]))
        except ValueError as error:
            raise SubmissionFailure("JSON submission is malformed") from error
        answer = value.get(ANSWER_FIELD) if isinstance(value, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise SubmissionFailure("JSON submission requires a nonempty string answer")
        return TextSubmission(answer)


class JsonValueAnswer(BaseAnswerFormat):
    """The complete final assistant message parsed as one JSON value."""

    kind: Literal["json_value"] = "json_value"
    submission_types = (JsonSubmission,)

    def supports(self, answer_type: AnswerType) -> bool:
        return answer_type == AnswerType.JSON

    def extract(self, attempt: GradingAttempt) -> JsonSubmission:
        try:
            return JsonSubmission(decode_json_value(_text_answer(attempt.conversation.events[-1])))
        except ValueError as error:
            raise SubmissionFailure("JSON value submission is malformed") from error


class AnswerCall(BaseAnswerFormat):
    """The string ``answer`` argument of one ``submit_answer`` function call."""

    kind: Literal["answer_call"] = "answer_call"
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


class FinalAction(BaseAnswerFormat):
    """The final assistant message itself: function calls to the task's final tools, or text."""

    kind: Literal["final_action"] = "final_action"
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


AnswerFormat = Annotated[
    PlainText | Boxed | JsonAnswer | JsonValueAnswer | AnswerCall | FinalAction, Field(discriminator="kind")
]


class TaskSpec(BaseModel):
    """The complete semantic definition of one task, its grader, and its final result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    context: ConversationInput
    environment_requirements: EnvironmentRequirements
    final_tools: tuple[FunctionDefinition, ...] = ()
    interaction_tools: tuple[FunctionDefinition, ...] = ()
    output_paths: tuple[str, ...] = ()
    answer_type: AnswerType
    answer_format: AnswerFormat
    grader: Grader
    source: Source
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
        if self.answer_type in CONVERSATION_ANSWERS and not self.answer_format.supports(self.answer_type):
            raise ValueError(f"Answer format {self.answer_format.kind} cannot carry a {self.answer_type} answer")
        validate_output_paths(self.output_paths)
        if grades_in_process(self.grader) and self.answer_type in (AnswerType.FILE, AnswerType.WORKSPACE_STATE):
            raise ValueError(f"A {self.answer_type} answer requires a grading environment")
        if isinstance(self.grader, VerifyitGrader) and self.grader.environment is not None:
            if self.answer_type == AnswerType.NATIVE_ACTION:
                raise ValueError("A verifyit grading environment cannot grade a native_action answer")
            if self.answer_type in CONVERSATION_ANSWERS and verifyit_answer_file(verifyit_spec(self.grader)) is None:
                raise ValueError(f"Workspace verifier {self.grader.mode} cannot grade a {self.answer_type} answer")
        if isinstance(self.grader, ScriptGrader):
            if self.grader.answer_path is not None and self.answer_type not in CONVERSATION_ANSWERS:
                raise ValueError(f"A {self.answer_type} answer has no extracted answer to write")
            installed = {f"/tests/{resource.path}" for resource in self.resources.verifier}
            if self.grader.conversation_path in installed:
                raise ValueError("The conversation path collides with a verifier resource")
        return self
