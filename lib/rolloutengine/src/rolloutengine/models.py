# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The core types: model requests and turns, rollout records and events, exceptions, and the task rollout spec."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from shellbox.machine import NetworkPolicy
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import TaskSpec
from verifyit.modes.grade_judge import JudgeConnection

LENGTH_STOP_REASON = "length"
MAX_TURNS_STOP_REASON = "max_turns"
TOTAL_TURN_TIMEOUT_STOP_REASON = "total_turn_timeout"
BUDGET_STOP_REASON = "budget_reached"


@dataclass(frozen=True)
class Continuation:
    """Tokens already served in this training record, which the next prompt must start with.

    The transport keeps them unchanged, so that it never tokenizes sampled text again.
    """

    served_token_ids: tuple[int, ...]
    """The prompt and sampled response tokens of the earlier turns, as served."""
    assistant_message_index: int
    """Where the served tokens end in ``messages``: the last assistant message."""


@dataclass(frozen=True)
class ModelRequest:
    """One request to the model transport, of one of three kinds.

    With neither ``continuation`` nor ``prompt_token_ids``, the transport renders all ``messages``. With
    ``continuation``, it renders only the messages after the served tokens and appends their tokens. With
    ``prompt_token_ids``, it serves those tokens unchanged, ``messages`` are only for display, and ``options``
    are sampling settings. A request never has both.
    """

    messages: tuple[dict[str, Any], ...]
    """The chat so far, one dict per message. New observations come last."""
    options: dict[str, Any]
    """Other request fields, such as tools and sampling settings. The engine does not read them."""
    continuation: Continuation | None = None
    """Tokens to keep unchanged. Tokenize only the messages after them."""
    prompt_token_ids: tuple[int, ...] | None = None
    """A full prompt that the harness pinned. Serve it unchanged."""


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
    """The transport's answer to one ``ModelRequest``: the parsed message and the exact tokens.

    Under ``TraceMode.MESSAGES``, the token fields can be empty and ``logprobs`` can be ``None``. The engine then
    records the token IDs without checking them. A chat endpoint that gives no token IDs sets ``usage``, so that
    ``RolloutLimits.max_total_tokens`` still counts its tokens.
    """

    message: dict[str, Any]
    """The assistant message that the transport parsed, for example with tool calls. The engine adds it to the chat."""
    prompt_token_ids: tuple[int, ...]
    """The exact prompt tokens that the model got. They start with the served tokens, or equal a pinned prompt."""
    response_token_ids: tuple[int, ...]
    """The tokens that the model sampled. Training uses them as the response."""
    logprobs: tuple[float, ...] | None
    """The log probability of each response token when it was sampled, or ``None`` if the transport gives none."""
    stop_reason: str
    """Why generation stopped, for example ``"stop"`` or ``"length"``. A ``"length"`` stop refuses the next ``reply``."""
    text: str = ""
    """The response as plain text. A harness that reads tags in the text, not tool calls, uses it."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra data from the transport, such as top-k log probabilities. The engine passes it on unread."""
    usage: TokenUsage | None = None
    """Token counts that the transport reports. The token budget counts them instead of the token IDs."""

    @property
    def request_tokens(self) -> int:
        """Tokens this request spent: the reported ``usage``, else the prompt and response token-ID counts."""
        if self.usage is not None:
            return self.usage.prompt_tokens + self.usage.completion_tokens
        return len(self.prompt_token_ids) + len(self.response_token_ids)


Model = Callable[[ModelRequest], Awaitable[ModelTurn]]
"""The policy under rollout: it serves one model request and returns the exact turn."""


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
class ConversationRecord:
    """One conversation of an attempt: from a ``ModelSession.start`` to the next one, or to the end."""

    messages: tuple[dict[str, Any], ...]
    """The chat of this conversation, through its last model response. The grader reads the last one with a turn."""
    prompt_token_ids: tuple[int, ...]
    """The tokens of the first prompt of this conversation."""
    response_token_ids: tuple[int, ...]
    """All tokens after the first prompt: each sampled response and the observation tokens between them."""
    loss_mask: tuple[int, ...]
    """One value per response token: 1 for a sampled token, 0 for an observation. All 0 in a messages trace."""
    logprobs: tuple[float, ...] | None
    """The sampling log probability of each response token, 0.0 for observation tokens. None if a turn had none."""
    steps: tuple["RolloutStep", ...] = ()
    """One step per model turn of this conversation: the turn, the transition, and where its tokens end."""


@dataclass(frozen=True)
class Rollout:
    """What happened in one attempt: its conversations, with their token evidence and steps, and the run outputs.

    Each ``ModelSession.start`` starts a new conversation, and the rollout keeps all of them. A trainer can use
    each conversation as a sample, or only the last one.
    """

    run_id: str
    """The ID of the attempt, the same as in its events."""
    conversations: tuple[ConversationRecord, ...]
    """One record for each conversation that the harness started, in order."""
    stop_reason: str
    """Why the attempt ended, for example ``"stop"``, ``"length"``, ``"max_turns"``, or ``"budget_reached"``."""
    metrics: dict[str, Any] = field(default_factory=dict)
    """The last step's metrics, and engine values such as the count of undelivered events."""
    outputs: dict[str, bytes] = field(default_factory=dict)
    """The files under ``TaskSpec.output_paths``, by path. The engine reads them after the harness ends."""
    failure: RolloutFailure | None = None
    """For an interrupted attempt: the exception type and the operation that failed."""


@dataclass(frozen=True)
class GradedRollout:
    """One attempt of a task, and its grade."""

    rollout: Rollout
    grade: GradeResult
    """The verdict of the task's grader on the last conversation that has a model turn. Unavailable if none has."""


class RolloutOperation(StrEnum):
    ATTEMPT = "attempt"
    START = "start"
    PREPARE = "prepare"
    MODEL = "model"
    ADVANCE = "advance"
    GRADE = "grade"


class RolloutInterrupted(RuntimeError):
    """An attempt that failed during ``operation``, with the rollout so far and the original exception as cause.

    ``PREPARE`` means that the harness failed before its first model turn, and ``ADVANCE`` that it failed after one.
    """

    def __init__(self, rollout: Rollout, operation: RolloutOperation):
        self.rollout = rollout
        self.operation = operation
        super().__init__(f"Rollout interrupted during {operation}")


class GradedRolloutInterrupted(RolloutInterrupted):
    """An interrupted task attempt, with its grade.

    ``grade`` grades the completed turns, because the engine still grades them when the model fails. It is
    unavailable when the attempt stopped before grading.
    """

    def __init__(self, rollout: Rollout, operation: RolloutOperation, grade: GradeResult):
        super().__init__(rollout, operation)
        self.grade = grade


class RolloutEnded(Exception):
    """The attempt permits no more model requests. After a failed model request, the model's error is the cause."""


class LimitReached(RolloutEnded):
    """A turn, token, or length limit ended the attempt. A harness lets it propagate, and the attempt ends normally."""

    def __init__(self, stop_reason: str):
        super().__init__(f"The rollout reached a limit: {stop_reason}")
        self.stop_reason = stop_reason


class RolloutContractError(ValueError):
    """Rollout evidence violates the exact-token contract."""


@dataclass(frozen=True)
class Transition:
    """What followed one model turn: observations for the next request, or the end of the harness."""

    done: bool
    """True when the harness ended after this turn."""
    observations: tuple[dict[str, Any], ...] = ()
    """New messages after the model's message, for example tool results. The last step's never reach the model."""
    token_ids: tuple[int, ...] | None = None
    """Pins the exact tokens of the next prompt. They must start with the tokens served so far."""
    metrics: dict[str, Any] = field(default_factory=dict)
    """The step's metrics. ``Rollout.metrics`` starts from the metrics of the last step."""
    data: object = None
    """The harness's own data for the step, for example rewards for a trainer. The engine does not read it."""


@dataclass(frozen=True)
class RolloutStep:
    turn: ModelTurn
    transition: Transition
    response_end: int
    """The index of the turn's last response token in the conversation's ``response_token_ids``."""
    messages: tuple[dict[str, Any], ...]
    """The chat at this step, through the turn's message."""


class RolloutEventType(StrEnum):
    """Names each event payload, so a serialized event keeps its type."""

    ATTEMPT_STARTED = "attempt_started"
    MACHINE_READY = "machine_ready"
    CONVERSATION_STARTED = "conversation_started"
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
class ConversationStarted:
    """The harness started a conversation with ``messages`` and ``options``, including its advertised tools.

    ``duration`` is the time from the start of the harness, or from the last model turn, to this request.
    """

    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    duration: float
    type: RolloutEventType = field(default=RolloutEventType.CONVERSATION_STARTED, init=False)


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
    """The harness handled a turn: it sent ``observations``, started a new conversation, or ended.

    ``duration`` is the time from the model turn to the harness's next request or its end.
    """

    turn_index: int
    done: bool
    observations: tuple[dict[str, Any], ...]
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
    | ConversationStarted
    | ModelTurnCompleted
    | ToolCallCompleted
    | TransitionCompleted
    | Graded
    | AttemptEnded
)


@dataclass(frozen=True)
class RolloutEvent:
    """One event of the attempt that ``RolloutEngine.run_task`` started with ``run_id`` for task ``task_id``."""

    run_id: str
    task_id: str
    payload: RolloutEventPayload


class RolloutObserver(Protocol):
    """Receive rollout progress in order, delivered off the rollout's path.

    The attempt never waits on delivery, so a slow or failing observer cannot change a rollout. After the
    attempt releases its resources, ``run_task`` waits up to ``cleanup_timeout`` for queued events.
    """

    async def on_event(self, event: RolloutEvent) -> None: ...


class EventSink(Protocol):
    """Queue an event payload for delivery as a ``RolloutEvent`` of the current attempt; never blocks."""

    def emit(self, payload: RolloutEventPayload) -> None: ...


class TraceMode(StrEnum):
    """The evidence a rollout records for each model turn."""

    EXACT_TOKENS = "exact_tokens"
    """Served token IDs, loss masks, and log probabilities under the exact-token contract."""
    MESSAGES = "messages"
    """Unchecked, informational token IDs with an all-zero loss mask; never train on these rollouts."""


TRACE_METRIC = "trace"
"""``Rollout.metrics`` key that marks every ``TraceMode.MESSAGES`` record with the mode's value."""


class MachineRuntimeSpec(BaseModel):
    """Deployment settings for one Shellbox machine."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    backend: str = Field(min_length=1)
    network: NetworkPolicy
    cpus: int | None = Field(gt=0)
    memory_mb: int | None = Field(gt=0)
    storage_mb: int | None = Field(gt=0)
    gpus: int = Field(ge=0)
    user: str | None
    startup_timeout: float | None = Field(gt=0)
    cleanup_timeout: float | None = Field(gt=0)


class RolloutLimits(BaseModel):
    """Turn, token, and time limits for one attempt.

    ``max_total_tokens`` stops model requests when the tokens spent, plus the known part of the next
    prompt, reach the threshold. The known part is the last request's prompt and response. The new
    observations of a request are not in that count, so the last request sent can exceed the threshold
    by its observation and response tokens. ``None`` sets no threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_turns: int = Field(gt=0)
    max_total_tokens: int | None = Field(gt=0)
    model_turn_timeout: float | None = Field(gt=0)
    tool_turn_timeout: float | None = Field(gt=0)
    total_turn_timeout: float | None = Field(gt=0)
    attempt_timeout: float | None = Field(gt=0)
    verifier_timeout: float | None = Field(gt=0)
    cleanup_timeout: float = Field(gt=0)


class TaskRolloutSpec(BaseModel):
    """An unchanged task definition and the choices that run it.

    ``harness`` names a harness registered with the engine. A machine selection of ``None``
    supplies no machine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskSpec
    harness: str = Field(min_length=1)
    task_machine: MachineRuntimeSpec | None
    verifier_machine: MachineRuntimeSpec | None
    judge_connection: JudgeConnection | None = Field(default=None, exclude=True)
    """The service of a judge grader, which grades on the host. ``model_dump``, JSON, and pickle leave it out,
    so a spec loaded from them has none. Without it, the engine reads ``VERIFYIT_JUDGE_BASE_URL`` and
    ``VERIFYIT_JUDGE_API_KEY``."""
    limits: RolloutLimits
    trace: TraceMode = TraceMode.EXACT_TOKENS

    def __getstate__(self) -> dict[Any, Any]:
        # Pickle leaves out the judge connection and its API key, as JSON does. The state holds the live fields.
        state = super().__getstate__()
        fields = {**state["__dict__"], "judge_connection": None}
        fields_set = state["__pydantic_fields_set__"] - {"judge_connection"}
        return {**state, "__dict__": fields, "__pydantic_fields_set__": fields_set}
