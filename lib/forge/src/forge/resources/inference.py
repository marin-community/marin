# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forge values at the boundary of the existing GLM transport and continuation engine."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, Protocol, override

from taskforge.llm.client import RETRYABLE_STATUSES, Attempt, AttemptOutcome, ModelRequestRejected, ModelUnavailable
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.policy import LLMPolicy, ReasoningEffort

from forge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, canonical_digest
from forge.effects.issues import Hold, Restart, Transient
from forge.model.llm import Completion, FinishReason, Message, Sampling, ToolCall, ToolSchema, Usage
from forge.model.serialization import TOOL_SCHEMA
from forge.resources.execution import Notify, retry_effect


class ContextExceeded(Exception):
    """The provider rejected the conversation because its context window is full."""


@dataclass(frozen=True, slots=True)
class InferenceRequest(Event[Completion]):
    result_type: ClassVar[object] = Completion
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...]
    sampling: Sampling


class InferenceClient(Protocol):
    def fingerprint(self) -> Fingerprint: ...
    async def complete(self, request: InferenceRequest) -> Completion: ...


class CompletionBackend(Protocol):
    """The subset of GlmClient used here, injectable without a network connection."""

    @property
    def model(self) -> str: ...
    async def complete(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> ProviderCompletion: ...


def _message(message: Message) -> dict[str, object]:
    wire: dict[str, object] = {"role": message.role, "content": message.content}
    if message.reasoning:
        wire["reasoning_content"] = message.reasoning
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        wire["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return wire


@dataclass(frozen=True, slots=True)
class GlmInference(InferenceClient):
    """Hold on non-retryable rejection: a human must inspect the request or provider configuration."""

    client: CompletionBackend
    provider_digest: str

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", canonical_digest((self.client.model, self.provider_digest)), ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        sampling = request.sampling
        settings = {key: value for key, value in asdict(sampling).items() if value is not None}
        settings["reasoning_effort"] = ReasoningEffort(sampling.reasoning_effort)
        policy = LLMPolicy(**settings)
        fields: dict[str, object] = {
            "tools": [
                {"type": "function", "function": TOOL_SCHEMA.dump_python(tool, mode="json")} for tool in request.tools
            ]
        }
        try:
            result = await self.client.complete(tuple(_message(message) for message in request.messages), policy, fields)
        except ModelUnavailable as error:
            raise Restart("GLM transport exhausted its retry or hold budget", _retries(error.attempts)) from error
        except ModelRequestRejected as error:
            if error.status == 400 and "maximum context length" in error.body.lower():
                raise ContextExceeded(error.body) from error
            decision = Transient if error.status in RETRYABLE_STATUSES else Hold
            detail = f": {error.body[:500]}" if decision is Hold else ""
            raise decision(
                f"GLM request rejected with status {error.status}{detail}", _retries(error.attempts)
            ) from error
        return Completion(
            result.content,
            tuple(ToolCall(call.id, call.name, call.arguments) for call in result.tool_calls),
            FinishReason(result.finish_reason),
            Usage(**asdict(result.usage)),
            result.continuations,
            result.reasoning,
            _retries(result.attempts),
        )


class FakeInference(InferenceClient):
    def __init__(
        self, completions: Sequence[Completion | Transient | Restart | Hold | ContextExceeded], *, identity: str = "fake"
    ) -> None:
        self.script = tuple(completions)
        self.requests: list[InferenceRequest] = []
        self.identity = identity

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", self.identity, ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self.script):
            raise RuntimeError("Fake inference script exhausted")
        result = self.script[index]
        if isinstance(result, Transient | Restart | Hold | ContextExceeded):
            raise result
        return result


def _retries(attempts: Sequence[Attempt]) -> tuple[str, ...]:
    return tuple(
        f"{attempt.outcome}:{attempt.http_status}"
        for attempt in attempts[:-1]
        if attempt.outcome != AttemptOutcome.COMPLETED
    )


async def complete(event: InferenceRequest, client: InferenceClient, *, notify: Notify) -> Completion:
    async def attempt() -> Completion:
        try:
            result = await client.complete(event)
        except (Transient, Restart, Hold) as error:
            for index, cause in enumerate(error.args[1] if len(error.args) > 1 else (), 1):
                notify(Retrying(event, index, str(cause)))
            raise
        for index, cause in enumerate(result.retries, 1):
            notify(Retrying(event, index, cause))
        return result

    return await retry_effect(event, attempt, notify)
