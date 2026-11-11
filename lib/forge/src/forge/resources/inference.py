# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forge values at the boundary of the existing GLM transport and continuation engine."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol, override

from taskforge.llm.agent import replay_arguments
from taskforge.llm.client import (
    RETRYABLE_STATUSES,
    Attempt,
    AttemptOutcome,
    ContextExhausted,
    ModelRequestRejected,
    ModelUnavailable,
)
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.policy import LLMPolicy

from forge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, digest
from forge.effects.issues import Hold, Restart, Transient
from forge.model.llm import TOOL_SCHEMA, Completion, FinishReason, Message, ToolCall, ToolSchema, Usage
from forge.resources.execution import Notify, retry_effect


class ContextExceeded(Exception):
    """The provider rejected the conversation because its context window is full."""


@dataclass(frozen=True, slots=True)
class InferenceRequest(Event[Completion]):
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...]
    sampling: LLMPolicy
    previous: "InferenceRequest | None" = None
    sample: int = 0
    _history: str = field(default="", init=False, repr=False, compare=False, metadata={"serialize": False})

    def __post_init__(self) -> None:
        history = self.previous._history if self.previous is not None else ""
        for message in self.messages:
            history = digest((history, message))
        object.__setattr__(self, "_history", history)

    @override
    def identity(self) -> object:
        return (self._history, self.tools, self.sampling) + ((self.sample,) if self.sample else ())

    @property
    def conversation(self) -> tuple[Message, ...]:
        turns = []
        turn: InferenceRequest | None = self
        while turn is not None:
            turns.append(turn.messages)
            turn = turn.previous
        return tuple(message for messages in reversed(turns) for message in messages)


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
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": replay_arguments(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return wire


@dataclass(frozen=True, slots=True)
class GlmInference(InferenceClient):
    """Translate retryable transport failures; invalid requests remain typed bugs."""

    client: CompletionBackend
    provider_digest: str

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", digest((self.client.model, self.provider_digest)), ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        policy = request.sampling
        fields: dict[str, object] = {
            "tools": [
                {"type": "function", "function": TOOL_SCHEMA.dump_python(tool, mode="json")} for tool in request.tools
            ]
        }
        # THROWAWAY (2026-11-09): with FORGE_ROUTING_KEY set, pin each session to one gateway worker by key.
        # The gateway reads ``prompt_cache_key`` from the body and forwards it as ``x-routing-key``; without it,
        # cache_aware routing sends every new session to the node holding the biggest shared-prefix cache.
        if os.environ.get("FORGE_ROUTING_KEY"):
            root = request
            while root.previous is not None:
                root = root.previous
            fields["prompt_cache_key"] = f"forge-{digest((self.provider_digest, root._history, root.sample))[:32]}"
        try:
            result = await self.client.complete(
                tuple(_message(message) for message in request.conversation), policy, fields
            )
        except ModelUnavailable as error:
            raise Restart(
                "GLM transport exhausted its retry or hold budget", retries=_retries(error.attempts)
            ) from error
        except ContextExhausted as error:
            raise ContextExceeded(error.body) from error
        except ModelRequestRejected as error:
            if error.status not in RETRYABLE_STATUSES:
                raise
            raise Transient(
                f"GLM request rejected with status {error.status}", retries=_retries(error.attempts)
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
            for index, cause in enumerate(error.retries, 1):
                notify(Retrying(event, index, str(cause)))
            raise
        for index, cause in enumerate(result.retries, 1):
            notify(Retrying(event, index, cause))
        return result

    return await retry_effect(event, attempt, notify)
