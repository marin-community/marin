# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forge values at the boundary of the existing GLM transport and continuation engine."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, Protocol, override

from taskforge.llm.client import RETRYABLE_STATUSES, Attempt, AttemptOutcome, ModelRequestRejected, ModelUnavailable
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.policy import LLMPolicy, ReasoningEffort

from forge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, canonical_digest
from forge.effects.issues import Hold, Restart, Transient
from forge.model.llm import Completion, FinishReason, Message, Sampling, ToolCall, ToolSchema, Usage
from forge.model.serialization import TOOL_SCHEMA


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
    client: CompletionBackend
    provider_digest: str
    max_attempts: int
    notify: Callable[[Retrying], Awaitable[None]]

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise Hold("Inference requires at least one attempt")

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", canonical_digest((self.client.model, self.provider_digest)), ReplayPolicy.CACHE)

    async def _attempts(self, request: InferenceRequest, attempts: Sequence[Attempt]) -> None:
        # GlmClient owns transport retries. Report them without retrying an exhausted logical call.
        for index, attempt in enumerate(attempts[:-1], 1):
            if attempt.outcome != AttemptOutcome.COMPLETED:
                await self.notify(Retrying(request, index, f"{attempt.outcome}:{attempt.http_status}"))

    async def _complete(self, request: InferenceRequest) -> Completion:
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
            await self._attempts(request, error.attempts)
            raise Restart("GLM transport exhausted its retry or hold budget") from error
        except ModelRequestRejected as error:
            await self._attempts(request, error.attempts)
            if error.status in RETRYABLE_STATUSES:
                raise Transient(f"GLM status {error.status}") from error
            raise Hold(f"GLM request rejected with status {error.status}") from error
        await self._attempts(request, result.attempts)
        return Completion(
            result.content,
            tuple(ToolCall(call.id, call.name, call.arguments) for call in result.tool_calls),
            FinishReason(result.finish_reason),
            Usage(**asdict(result.usage)),
            result.continuations,
            result.reasoning,
        )

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await self._complete(request)
            except Transient as error:
                if attempt == self.max_attempts:
                    raise Restart("Inference executor exhausted its retry budget") from error
                await self.notify(Retrying(request, attempt, str(error)))
        raise Restart("Inference executor made no attempt")


class FakeInference(InferenceClient):
    def __init__(self, completions: Sequence[Completion | Restart | Hold], *, identity: str = "fake") -> None:
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
            raise Restart("Fake inference script exhausted")
        result = self.script[index]
        if isinstance(result, Restart | Hold):
            raise result
        return result
