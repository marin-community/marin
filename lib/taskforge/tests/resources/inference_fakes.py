# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scripted completion backend, a scripted provider and a request shared by the inference and provider tests."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from taskforge.llm.client import (
    Attempt,
    AttemptFailed,
    AttemptOutcome,
    ModelRequestRejected,
    ModelUnavailable,
    Progress,
    Provider,
    Segment,
)
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.policy import GLM_5_3, LLMPolicy
from taskforge.model.llm import FinishReason, Message, Role, ToolCall, ToolSchema, Usage
from taskforge.resources.inference import CompletionBackend, InferenceRequest


class Backend(CompletionBackend):
    model = "test-model"
    profile = GLM_5_3

    def __init__(self, script: Sequence[ProviderCompletion | ModelRequestRejected | ModelUnavailable]) -> None:
        self.script = tuple(script)
        self.calls: list[tuple[Sequence[dict[str, object]], LLMPolicy, Mapping[str, object] | None]] = []

    @override
    async def complete(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> ProviderCompletion:
        value = self.script[len(self.calls)]
        self.calls.append((messages, policy, request_fields))
        if isinstance(value, ModelRequestRejected | ModelUnavailable):
            raise value
        return value


def provider_completion(attempts: tuple[Attempt, ...] = ()) -> ProviderCompletion:
    return ProviderCompletion(
        "answer",
        "reason",
        (ToolCall("id", "shell", "{}"),),
        FinishReason.TOOL_CALLS,
        Usage(10, 20, 3, 4),
        1,
        0.1,
        0.9,
        2,
        attempts,
    )


def request() -> InferenceRequest:
    return InferenceRequest(
        (
            Message(Role.ASSISTANT, "prefix", (ToolCall("previous", "shell", "{}"),), reasoning="thought"),
            Message(Role.TOOL, "result", tool_call_id="previous"),
        ),
        (ToolSchema("shell", "Run command", {"type": "object"}),),
        LLMPolicy(max_tokens=100),
    )


def segment(
    content: str = "", tool_calls: tuple[ToolCall, ...] = (), finish: FinishReason = FinishReason.STOP
) -> Segment:
    return Segment(content, "", tool_calls, finish, Usage(10, 5), 0.0, 0.0, 0.0)


def chunk(**delta: object) -> str:
    """One streamed ``data:`` payload carrying ``delta``."""
    return json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})


@dataclass(frozen=True)
class Fails:
    """A failed attempt that received ``events`` first."""

    outcome: AttemptOutcome
    events: tuple[str, ...] = ()
    status: int | None = None
    detail: str = "scripted"


HANG = "hang"
"""A healthy stream that never finishes."""


class LoopTime:
    """A fake clock that is also the running event loop's: ``sleep`` jumps both forward.

    Loop deadlines, such as RolloutEngine's model-turn timeout, then expire at their virtual time while the test runs in
    milliseconds. Construct it inside the test's event loop.
    """

    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.real, self.offset = loop.time, 0.0
        loop.time = self.clock  # type: ignore[method-assign]

    def clock(self) -> float:
        return self.real() + self.offset

    async def sleep(self, delay: float) -> None:
        self.offset += delay
        await asyncio.sleep(0)


def generated(payload: str) -> bool:
    """Whether a payload carries generated text, as ``GlmProvider`` decides it."""
    deltas = [choice.get("delta") or {} for choice in json.loads(payload).get("choices") or ()]
    return any(delta.get("content") or delta.get("reasoning") or delta.get("tool_calls") for delta in deltas)


def report(progress: Progress, payload: str) -> None:
    progress.event(payload, generated(payload))


Act = Callable[[Progress], Awaitable[Segment]]
"""A scripted attempt that drives its own events and time."""


class ScriptedProvider(Provider):
    """Answers each attempt with the next script item; ``empty_pool`` answers each availability poll in turn."""

    def __init__(
        self, script: Sequence[Segment | Fails | str | Act], model: str = "test-model", empty_pool: Sequence[bool] = ()
    ) -> None:
        self.script, self.name, self.empty_pool = tuple(script), model, list(empty_pool)
        self.bodies: list[Mapping[str, Any]] = []
        self.polls = 0
        self.streaming = asyncio.Event()

    @property
    @override
    def model(self) -> str:
        return self.name

    @override
    async def stream(self, body: Mapping[str, Any], progress: Progress) -> Segment:
        item = self.script[len(self.bodies)]
        self.bodies.append(body)
        self.streaming.set()
        if item == HANG:
            await asyncio.Event().wait()
        if isinstance(item, Fails):
            for payload in item.events:
                report(progress, payload)
            raise AttemptFailed(item.outcome, item.detail, item.status)
        if callable(item):
            return await item(progress)
        assert isinstance(item, Segment)
        return item

    @override
    async def no_workers(self) -> bool:
        self.polls += 1
        return self.empty_pool.pop(0) if self.empty_pool else False
