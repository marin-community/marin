# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public rollout protocols, records, and exceptions."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome

LENGTH_STOP_REASON = "length"
MAX_TURNS_STOP_REASON = "max_turns"
TOTAL_TURN_TIMEOUT_STOP_REASON = "total_turn_timeout"
BUDGET_STOP_REASON = "budget_reached"


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
class Prompt:
    """Messages and model options for a request; ``token_ids`` pins the exact prompt tokens to serve."""

    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts that the model transport reports for one inference request."""

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ModelTurn:
    """Exact tokens and the parsed message from one inference request.

    Under ``TraceMode.MESSAGES`` the token fields may be empty and ``logprobs`` may be ``None``;
    the engine records reported token IDs without checking them.
    ``usage`` overrides the token-ID counts when the engine charges this request
    against ``RolloutLimits.max_total_tokens``. Transports that report usage but
    not exact token IDs, such as chat endpoints, set it so the budget still counts.
    """

    message: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    stop_reason: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage | None = None

    @property
    def request_tokens(self) -> int:
        """Tokens this request spent: the reported ``usage``, else the prompt and response token-ID counts."""
        if self.usage is not None:
            return self.usage.prompt_tokens + self.usage.completion_tokens
        return len(self.prompt_token_ids) + len(self.response_token_ids)


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
    """A harness's response to one model turn.

    ``observations`` extend the conversation for the next request, and ``token_ids`` can pin the exact
    prompt tokens of that request. ``reset`` replaces the conversation with a new prompt instead, and the
    engine starts a new training record. ``data`` is the harness's own record of the step, for example
    the rewards that a trainer reads; the engine keeps it unchanged in ``RolloutStep.transition``.
    """

    done: bool
    observations: tuple[dict[str, Any], ...] = ()
    token_ids: tuple[int, ...] | None = None
    reset: Prompt | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    data: object = None


@dataclass(frozen=True)
class RolloutStep:
    turn: ModelTurn
    transition: Transition
    response_end: int
    messages: tuple[dict[str, Any], ...]


class RolloutEventType(StrEnum):
    """Names each event payload, so a serialized event keeps its type."""

    ATTEMPT_STARTED = "attempt_started"
    MACHINE_READY = "machine_ready"
    HARNESS_PREPARED = "harness_prepared"
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
class HarnessPrepared:
    """The harness returned the first request's ``messages`` and ``options``, including its advertised tools."""

    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.HARNESS_PREPARED, init=False)


@dataclass(frozen=True)
class ModelTurnCompleted:
    """A model turn passed the token checks; ``message`` is the assistant message it produced."""

    turn_index: int
    message: dict[str, Any]
    stop_reason: str
    prompt_token_count: int
    response_token_count: int
    usage: TokenUsage | None
    metadata: dict[str, Any]
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.MODEL_TURN_COMPLETED, init=False)


@dataclass(frozen=True)
class ToolCallCompleted:
    """A harness tool call returned an observation; ``done`` means the call ended the rollout.

    The observation itself arrives in the turn's ``TransitionCompleted``, keyed by ``call_id``.
    """

    call_id: str
    name: str
    observation_bytes: int
    done: bool
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.TOOL_CALL_COMPLETED, init=False)


@dataclass(frozen=True)
class TransitionCompleted:
    """The harness handled a turn and returned ``observations``, or a new prompt in ``reset``.

    The engine skips a reset when the turn or token limit ends the rollout.
    """

    turn_index: int
    done: bool
    observations: tuple[dict[str, Any], ...]
    reset: Prompt | None
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.TRANSITION_COMPLETED, init=False)


@dataclass(frozen=True)
class Graded:
    status: Outcome
    reward: float | None
    error: str | None
    failure: GradingFailure | None
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.GRADED, init=False)


@dataclass(frozen=True)
class AttemptEnded:
    """The attempt returned with ``stop_reason``, or was interrupted during ``operation`` by ``error``.

    ``duration`` covers the whole attempt, including resource cleanup.
    """

    stop_reason: str | None
    operation: RolloutOperation | None
    error: str | None
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.ATTEMPT_ENDED, init=False)


RolloutEventPayload = (
    AttemptStarted
    | MachineReady
    | HarnessPrepared
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
