# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The model surface: call records, the errors a call raises, and the ``ModelClient`` protocol.

Every logical call is a ``Completion``: the concatenation of its continuation segments, with the
``Attempt`` record of every HTTP request it made.

``ModelClient`` is the one interface callers reach a model through: free-text ``complete`` and
``complete_structured``, which returns a reply validated as a pydantic model with every completion
the call took. ``taskforge.llm.glm_client.GlmClient`` implements it.

The errors here are the ``ModelClient`` protocol's, and any implementation raises them:
``ModelRequestRejected`` is a rejection a retry cannot fix, ``ContextExhausted`` a prompt too long
for the model's context window, and ``ModelUnavailable`` retries or the infrastructure hold
running out.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from taskforge.llm.policy import LLMPolicy, Message


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ABORT = "abort"
    """vLLM aborted the request (for example on worker drain); the attempt is retried."""


class AttemptOutcome(StrEnum):
    COMPLETED = "completed"
    RETRYABLE_STATUS = "retryable_status"
    ROUTE_MISSING = "route_missing"
    CONTEXT_OVERFLOW = "context_overflow"
    TRANSPORT_ERROR = "transport_error"
    STALLED = "stalled"
    INCOMPLETE_STREAM = "incomplete_stream"
    STREAM_ERROR = "stream_error"
    ABORTED = "aborted"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass(frozen=True)
class Attempt:
    """One HTTP request, successful or not, with the raw SSE ``data:`` payloads it received.

    ``segment`` is the continuation index.
    """

    segment: int
    outcome: AttemptOutcome
    max_tokens: int
    started: float
    duration: float
    http_status: int | None
    detail: str
    events: tuple[str, ...]


@dataclass(frozen=True)
class Completion:
    """One logical call: the concatenation of its continuation segments.

    ``ttft`` is measured on the first segment's successful attempt; ``decode_time`` sums, over each
    segment's successful attempt, the time after its first token. ``wall_time`` includes backoff.
    """

    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    wall_time: float
    ttft: float
    decode_time: float
    continuations: int
    attempts: tuple[Attempt, ...]


class ModelRequestRejected(Exception):
    """The server rejected the request in a way a retry cannot fix."""

    def __init__(self, status: int, body: str):
        super().__init__(f"model server rejected request with HTTP {status}: {body}")
        self.status = status
        self.body = body


class ContextExhausted(ModelRequestRejected):
    """The prompt plus the requested output does not fit the model's context window."""

    def __init__(self, body: str):
        super().__init__(400, body)


class ModelUnavailable(Exception):
    """Retries or the infrastructure hold ran out; ``attempts`` holds the record."""

    def __init__(self, message: str, attempts: Sequence[Attempt]):
        super().__init__(message)
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class StructuredResult[T]:
    """A structured call's validated value and every completion it took, in order (a repair request adds one)."""

    value: T
    completions: tuple[Completion, ...]


class ModelClient(Protocol):
    """A model: ``model`` names it, ``complete`` samples a reply, ``complete_structured`` returns a
    reply that ``output_type`` validates.

    Every implementation raises the errors of this module: ``ModelUnavailable`` once transient
    failures run out, ``ContextExhausted`` for a prompt that does not fit the context window, and
    ``ModelRequestRejected`` for any other request no retry can fix.
    """

    @property
    def model(self) -> str: ...

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy) -> Completion: ...

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]: ...
