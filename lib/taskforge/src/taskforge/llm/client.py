# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Records of model calls, the ``ModelClient`` protocol, and the errors its implementations raise.

A logical call returns a ``Completion``, which joins the call's continuation segments and holds an
``Attempt`` for every HTTP request the call made. The error from a failed call carries the attempts
made up to that point.

Callers go through ``ModelClient`` for free-text replies (``complete``) and for replies validated
against a pydantic output type (``complete_structured``, which also returns every completion the call took).
``taskforge.llm.glm_client.GlmClient`` implements it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from taskforge.llm.policy import LLMPolicy, Message

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
"""HTTP statuses a retry can cure: a timeout, rate limiting or a server-side failure."""


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ABORT = "abort"
    """vLLM aborted the request, for example while draining a worker. The client retries the attempt."""


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
    REJECTED = "rejected"


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
    """An HTTP request, successful or failed, with the raw SSE ``data:`` payloads it received.

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
    """A logical call, joined from its continuation segments.

    ``ttft`` comes from the successful attempt of the first segment. ``decode_time`` adds up the time
    after the first token in each segment's successful attempt. ``wall_time`` includes backoff.
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
    """The server rejected a request for a reason a retry cannot fix; ``attempts`` ends with that request."""

    def __init__(self, status: int, body: str, attempts: Sequence[Attempt]):
        super().__init__(f"model server rejected request with HTTP {status}: {body}")
        self.status = status
        self.body = body
        self.attempts = tuple(attempts)


class ContextExhausted(ModelRequestRejected):
    """The prompt plus the requested output exceeds the model's context window."""

    def __init__(self, body: str, attempts: Sequence[Attempt]):
        super().__init__(400, body, attempts)


class ModelUnavailable(Exception):
    """The retries or the infrastructure hold ran out; ``attempts`` holds every attempt made."""

    def __init__(self, message: str, attempts: Sequence[Attempt]):
        super().__init__(message)
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class StructuredResult[T]:
    """The validated value of a structured call, with its completions in order; each repair adds a completion."""

    value: T
    completions: tuple[Completion, ...]


class ModelClient(Protocol):
    """Protocol for calling a model.

    ``model`` is the model id. ``complete`` samples a free-text reply, with ``request_fields`` such as
    ``tools`` added to the request body. ``complete_structured`` returns a reply validated as
    ``output_type``. Implementations raise this module's errors:

    * ``ModelUnavailable`` once the retries or the infrastructure hold run out.
    * ``ContextExhausted`` for a prompt too long for the model's context window.
    * ``ModelRequestRejected`` for any other rejection a retry cannot fix.
    """

    @property
    def model(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> Completion: ...

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]: ...
