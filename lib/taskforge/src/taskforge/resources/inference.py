# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded inference requests and their executor in front of the model client."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, override

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, digest
from taskforge.effects.issues import ContextExceeded, Hold, Restart, Transient
from taskforge.llm.client import Attempt, AttemptOutcome, ContextExhausted, ModelUnavailable
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.policy import LLMPolicy, ModelProfile
from taskforge.llm.wire import wire_tool_call
from taskforge.model.llm import TOOL_SCHEMA, Completion, Message, ToolSchema
from taskforge.resources.execution import Notify


@dataclass(frozen=True, slots=True)
class InferenceRequest(Event[Completion]):
    kind = "inference.request"
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
    """The subset of ``ModelClient`` used here, injectable without a network connection."""

    @property
    def model(self) -> str: ...
    @property
    def profile(self) -> ModelProfile: ...
    async def complete(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> ProviderCompletion: ...


def _message(message: Message) -> dict[str, object]:
    """``message`` on the wire; the harness already shaped its tool-call arguments with the profile's formatter."""
    wire: dict[str, object] = {"role": message.role, "content": message.content}
    if message.reasoning:
        wire["reasoning_content"] = message.reasoning
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        wire["tool_calls"] = [wire_tool_call(call) for call in message.tool_calls]
    return wire


@dataclass(frozen=True, slots=True)
class GlmInference(InferenceClient):
    """Translate an unavailable provider into a stage restart; other rejections remain typed bugs."""

    client: CompletionBackend
    role: str

    @override
    def fingerprint(self) -> Fingerprint:
        profile = self.client.profile.replay_identity()
        identity = (self.client.model, self.role) + (() if profile is None else (profile,))
        return Fingerprint("inference", digest(identity), ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        policy = request.sampling
        fields: dict[str, object] = {
            "tools": [
                {"type": "function", "function": TOOL_SCHEMA.dump_python(tool, mode="json")} for tool in request.tools
            ]
        }
        try:
            result = await self.client.complete(tuple(map(_message, request.conversation)), policy, fields)
        except ModelUnavailable as error:
            raise Restart(str(error), retries=_retries(error.attempts)) from error
        except ContextExhausted as error:
            raise ContextExceeded(error.body, retries=_retries(error.attempts)) from error
        return Completion(
            result.content,
            result.tool_calls,
            result.finish_reason,
            result.usage,
            result.continuations,
            result.reasoning,
            _retries(result.attempts),
        )


def _retries(attempts: Sequence[Attempt]) -> tuple[str, ...]:
    return tuple(
        f"{attempt.outcome}:{attempt.http_status}"
        for attempt in attempts[:-1]
        if attempt.outcome != AttemptOutcome.COMPLETED
    )


async def complete(event: InferenceRequest, client: InferenceClient, *, notify: Notify) -> Completion:
    """Run ``event`` once: the model client owns its retries, which are reported as ``Retrying`` notes."""
    try:
        result = await client.complete(event)
    except (Transient, Restart, Hold, ContextExceeded) as error:
        for index, cause in enumerate(error.retries, 1):
            notify(Retrying(event, index, str(cause)))
        raise
    for index, cause in enumerate(result.retries, 1):
        notify(Retrying(event, index, cause))
    return result
