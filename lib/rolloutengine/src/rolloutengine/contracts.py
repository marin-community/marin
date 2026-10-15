# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public rollout protocols, records, and exceptions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from shellbox.machine import Machine
from taskcompendium.grading_result import GradeResult, Outcome

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

    async def grade(self, messages: tuple[dict[str, Any], ...], *, stop_reason: str) -> GradeResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class AttemptStarted:
    task_id: str


@dataclass(frozen=True)
class MachineReady:
    """The task machine is created, its resources installed, and its setup commands run."""

    task_id: str
    duration: float


@dataclass(frozen=True)
class SessionPrepared:
    task_id: str
    message_count: int
    duration: float


@dataclass(frozen=True)
class ModelTurnCompleted:
    task_id: str
    turn_index: int
    stop_reason: str
    prompt_token_count: int
    response_token_count: int
    metadata: dict[str, Any]
    duration: float


@dataclass(frozen=True)
class ToolCallCompleted:
    """A session tool call returned an observation to the model."""

    task_id: str
    name: str
    observation_bytes: int
    duration: float


@dataclass(frozen=True)
class TransitionCompleted:
    task_id: str
    turn_index: int
    done: bool
    observation_count: int
    duration: float


@dataclass(frozen=True)
class Graded:
    task_id: str
    status: Outcome
    reward: float | None


@dataclass(frozen=True)
class AttemptEnded:
    """The attempt returned with ``stop_reason``, or was interrupted during ``operation``."""

    task_id: str
    stop_reason: str | None
    operation: RolloutOperation | None


RolloutEvent = (
    AttemptStarted
    | MachineReady
    | SessionPrepared
    | ModelTurnCompleted
    | ToolCallCompleted
    | TransitionCompleted
    | Graded
    | AttemptEnded
)


class RolloutObserver(Protocol):
    """Receive rollout progress; an exception raised here fails the attempt."""

    async def on_event(self, event: RolloutEvent) -> None: ...


async def emit_event(observer: RolloutObserver | None, event: RolloutEvent) -> None:
    if observer is not None:
        await observer.on_event(event)


class TaskGrader(Protocol):
    async def __call__(self, messages: tuple[dict[str, Any], ...], *, stop_reason: str) -> GradeResult: ...


@dataclass(frozen=True)
class TaskSessionContext:
    """Inputs for one task session; session-graded tasks have no task grader.

    A session may report its own events, such as tool calls, to ``observer``.
    """

    lowered: LoweredTaskSpec
    machine: Machine | None
    task_grader: TaskGrader | None
    observer: RolloutObserver | None = None

    def require_task_grader(self) -> TaskGrader:
        """Return the task grader for a session that delegates final grading."""
        if self.task_grader is None:
            raise ValueError("This task uses session-owned grading")
        return self.task_grader


TaskSessionFactory = Callable[[TaskSessionContext], TaskSession]
