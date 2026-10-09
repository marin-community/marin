# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The GLM-5.3 call records: endpoint, completion, usage and attempt types, and the errors a call raises.

Every logical call is a ``Completion``: the concatenation of its continuation segments, with the
``Attempt`` record of every HTTP request it made. ``GlmRequestRejected`` is a rejection a retry
cannot fix, ``GlmContextExhausted`` a prompt that fills the context window, and ``GlmUnavailable``
retries or the infrastructure hold running out.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from taskforge.llm.endpoint import API_ROOT, GLM_MODEL


class Pool(StrEnum):
    """Router worker pool; the bearer token binds which one serves a request."""

    HIGH = "high"
    BULK = "bulk"


@dataclass(frozen=True)
class GlmEndpoint:
    """An OpenAI-compatible base URL ending in ``/v1``, its bearer token, and the pool it serves."""

    base_url: str
    token: str = field(repr=False)
    pool: Pool
    model: str = GLM_MODEL

    def __post_init__(self) -> None:
        if not self.base_url.endswith(API_ROOT):
            raise ValueError(f"base_url must end in {API_ROOT}: {self.base_url}")


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

    ``segment`` is the continuation index; -1 marks the one-token probe that measures a prompt
    after a context-overflow rejection.
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

    @property
    def decode_tokens_per_second(self) -> float:
        return self.usage.completion_tokens / self.decode_time


class GlmRequestRejected(Exception):
    """The server rejected the request in a way a retry cannot fix."""

    def __init__(self, status: int, body: str):
        super().__init__(f"GLM rejected request with HTTP {status}: {body}")
        self.status = status
        self.body = body


class GlmContextExhausted(GlmRequestRejected):
    """The prompt alone fills the context window, so no output token fits."""

    def __init__(self, body: str):
        super().__init__(400, body)


class GlmUnavailable(Exception):
    """Retries or the infrastructure hold ran out; ``attempts`` holds the record."""

    def __init__(self, message: str, attempts: Sequence[Attempt]):
        super().__init__(message)
        self.attempts = tuple(attempts)
