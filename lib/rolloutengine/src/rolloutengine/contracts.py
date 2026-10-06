# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public rollout protocols, records, and exceptions."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from taskcompendium.environment import EnvironmentCommand, EnvironmentFile
from taskcompendium.grading_result import GradeResult

LENGTH_STOP_REASON = "length"
MAX_TURNS_STOP_REASON = "max_turns"
AGENT_TIMEOUT_STOP_REASON = "agent_timeout"


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    prefix_token_ids: tuple[int, ...]
    assistant_message_index: int | None


@dataclass(frozen=True)
class SessionStart:
    """The initial conversation and model options for one task session."""

    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    """Exact tokens and the parsed message from one inference request.

    ``stop_reason`` is ``LENGTH_STOP_REASON`` whenever generation ended on the output token budget,
    including a reply that a server's tool parser reports as tool calls. The default session does
    not execute tool calls from such a turn.
    """

    message: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    stop_reason: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationLimitReached(Exception):
    """The rendered prompt leaves no permitted generation budget.

    ``prompt_token_ids`` are the rendered prompt tokens. An adapter whose server rejects an
    over-limit prompt without rendering it passes ``ModelRequest.prefix_token_ids`` instead, which
    is empty on the first request. The engine uses these tokens only when no turn completed.
    """

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
    STATE = "state"
    GRADE = "grade"
    CLEANUP = "cleanup"


@dataclass(frozen=True)
class SuppliedState:
    """A final task state that `ShellboxRolloutEngine.grade_state` grades without model inference.

    Attributes:
        messages: The exact conversation the verifier receives, ending with the submission when
            the verifier reads one. A shell verifier that reads only files may receive none.
        files: Files installed at absolute machine paths after environment and stage setup.
        commands: Commands run in order after the files, for state that is not a file.
    """

    messages: tuple[dict[str, Any], ...] = ()
    files: tuple[EnvironmentFile, ...] = ()
    commands: tuple[EnvironmentCommand, ...] = ()


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


@dataclass(frozen=True)
class RolloutStep:
    turn: ModelTurn
    transition: Transition
    response_end: int
    messages: tuple[dict[str, Any], ...]


class TaskSession(Protocol):
    """Task operations without model inference or token processing."""

    async def prepare(self) -> SessionStart: ...

    async def advance(self, turn: ModelTurn) -> Transition: ...

    async def grade(self, messages: tuple[dict[str, Any], ...]) -> GradeResult: ...

    async def close(self) -> None: ...
