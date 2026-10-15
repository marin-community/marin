# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public rollout protocols, records, and exceptions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import JsonValue
from shellbox.machine import Machine
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from rolloutengine.spec import LoweredTaskSpec

LENGTH_STOP_REASON = "length"
MAX_TURNS_STOP_REASON = "max_turns"
TOTAL_TURN_TIMEOUT_STOP_REASON = "total_turn_timeout"


@dataclass(frozen=True)
class ModelRequest:
    """Request chat generation or an exact token-prefix replay.

    When prompt_token_ids is set, messages are retained for display only and
    options are completion sampling options, not chat-template options.
    """

    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    prefix_token_ids: tuple[int, ...]
    assistant_message_index: int | None
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SessionStart:
    """Initial conversation, options, and an optional exact token-prefix replay."""

    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ModelTurn:
    """Exact tokens and the parsed message from one inference request."""

    message: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    stop_reason: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationLimitReached(Exception):
    """The rendered prompt leaves no permitted generation budget."""

    def __init__(self, prompt_token_ids: tuple[int, ...]):
        self.prompt_token_ids = prompt_token_ids
        super().__init__("The prompt reached the configured generation limit")


@dataclass(frozen=True)
class RolloutFailure:
    """Safe failure fields for consumers of a rollout record."""

    exception_type: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutData:
    """One attempt's conversation, token evidence, grade, and captured run outputs.

    ``outputs`` maps each regular file under ``TaskSpec.output_paths`` on the task machine
    to its bytes. The engine captures them once after the turn loop, before final grading.
    """

    task_id: str
    messages: tuple[dict[str, Any], ...]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    grade: GradeResult
    stop_reason: str
    steps: tuple["RolloutStep", ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, bytes] = field(default_factory=dict)
    failure: RolloutFailure | None = None


class RolloutOperation(StrEnum):
    ATTEMPT = "attempt"
    START = "start"
    PREPARE = "prepare"
    MODEL = "model"
    ADVANCE = "advance"
    GRADE = "grade"
    CLEANUP = "cleanup"


class RolloutInterrupted(RuntimeError):
    """An execution failure with the last completed rollout and original exception cause."""

    def __init__(self, rollout: RolloutData, operation: RolloutOperation):
        self.rollout = rollout
        self.operation = operation
        super().__init__(f"Rollout interrupted during {operation}")


class RolloutContractError(ValueError):
    """Rollout evidence violates the exact-token contract."""


@dataclass(frozen=True)
class Transition:
    """One task transition after a model response."""

    done: bool
    observations: tuple[dict[str, Any], ...] = ()
    reset_conversation: tuple[dict[str, Any], ...] | None = None
    reward: float | None = None
    token_rewards: tuple[float, ...] | None = None
    token_credit: tuple[float, ...] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    grade: GradeResult | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    next_options: dict[str, Any] | None = None
    next_prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RolloutStep:
    turn: ModelTurn
    transition: Transition
    response_end: int
    messages: tuple[dict[str, Any], ...]


class TaskSession(Protocol):
    """Prepare model requests, handle model turns, and grade one task attempt."""

    async def prepare(self) -> SessionStart: ...

    async def advance(self, turn: ModelTurn) -> Transition: ...

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        """Grade the attempt; ``outputs`` holds the run outputs the engine captured after the turn loop."""
        ...

    async def close(self) -> None: ...


class RolloutEventType(StrEnum):
    """Names each event payload, so a serialized event keeps its type."""

    ATTEMPT_STARTED = "attempt_started"
    MACHINE_READY = "machine_ready"
    SESSION_PREPARED = "session_prepared"
    MODEL_TURN_COMPLETED = "model_turn_completed"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TRANSITION_COMPLETED = "transition_completed"
    GRADED = "graded"
    ATTEMPT_ENDED = "attempt_ended"


@dataclass(frozen=True)
class AttemptStarted:
    type: RolloutEventType = field(default=RolloutEventType.ATTEMPT_STARTED, init=False)


@dataclass(frozen=True)
class MachineReady:
    """The task machine is created, its resources installed, and its setup commands run."""

    duration: float
    type: RolloutEventType = field(default=RolloutEventType.MACHINE_READY, init=False)


@dataclass(frozen=True)
class SessionPrepared:
    message_count: int
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.SESSION_PREPARED, init=False)


@dataclass(frozen=True)
class ModelTurnCompleted:
    turn_index: int
    stop_reason: str
    prompt_token_count: int
    response_token_count: int
    metadata: dict[str, Any]
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.MODEL_TURN_COMPLETED, init=False)


@dataclass(frozen=True)
class ToolCallCompleted:
    """A session tool call returned an observation to the model."""

    name: str
    observation_bytes: int
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.TOOL_CALL_COMPLETED, init=False)


@dataclass(frozen=True)
class TransitionCompleted:
    turn_index: int
    done: bool
    observation_count: int
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.TRANSITION_COMPLETED, init=False)


@dataclass(frozen=True)
class Graded:
    status: Outcome
    reward: float | None
    type: RolloutEventType = field(default=RolloutEventType.GRADED, init=False)


@dataclass(frozen=True)
class AttemptEnded:
    """The attempt returned with ``stop_reason``, or was interrupted during ``operation``."""

    stop_reason: str | None
    operation: RolloutOperation | None
    type: RolloutEventType = field(default=RolloutEventType.ATTEMPT_ENDED, init=False)


RolloutEventPayload = (
    AttemptStarted
    | MachineReady
    | SessionPrepared
    | ModelTurnCompleted
    | ToolCallCompleted
    | TransitionCompleted
    | Graded
    | AttemptEnded
)


@dataclass(frozen=True)
class RolloutEvent:
    """One event of the attempt that ``RolloutEngine.run`` started with ``run_key`` for task ``task_id``."""

    run_key: str
    task_id: str
    payload: RolloutEventPayload


class RolloutObserver(Protocol):
    """Receive rollout progress in order, delivered off the rollout's path.

    The attempt never waits on delivery, so a slow or failing observer cannot change a rollout. After the
    attempt releases its resources, ``run`` waits up to ``cleanup_timeout`` for queued events.
    """

    async def on_event(self, event: RolloutEvent) -> None: ...


class EventSink(Protocol):
    """Queue an event payload for delivery as a ``RolloutEvent`` of the current attempt; never blocks."""

    def emit(self, payload: RolloutEventPayload) -> None: ...


class TaskGrader(Protocol):
    async def __call__(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult: ...


@dataclass(frozen=True)
class TaskSessionContext:
    """Inputs for one task session; session-graded tasks have no task grader.

    A session may report its own events, such as tool calls, to ``events``.
    """

    lowered: LoweredTaskSpec
    machine: Machine | None
    task_grader: TaskGrader | None
    events: EventSink

    def require_task_grader(self) -> TaskGrader:
        """Return the task grader for a session that delegates final grading."""
        if self.task_grader is None:
            raise ValueError("This task uses session-owned grading")
        return self.task_grader


TaskSessionFactory = Callable[[TaskSessionContext], TaskSession]


@dataclass(frozen=True)
class ToolContext:
    """The attempt's task machine and the task requirement a provider session serves."""

    machine: Machine | None
    requirement: ProviderRequirement


@dataclass(frozen=True)
class ProviderResult:
    """A tool observation; ``done`` ends the rollout after this call, and later calls of the turn do not run."""

    content: str
    done: bool = False


class ProviderSession(Protocol):
    """One attempt's provider state for one task requirement."""

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ProviderResult: ...

    async def close(self) -> None: ...


class ToolProvider(Protocol):
    """Host tools for one action interface, served beside the built-in shell tool."""

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        """The functions advertised to the model for one task requirement."""
        ...

    async def open(self, context: ToolContext) -> ProviderSession:
        """Start the state of one attempt, typically from ``context.requirement.initial_state``."""
        ...
