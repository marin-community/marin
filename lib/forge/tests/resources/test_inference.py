# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import override

import pytest
from taskforge.llm.client import Attempt, AttemptOutcome, ModelRequestRejected, ModelUnavailable
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.client import FinishReason as ProviderFinish
from taskforge.llm.client import ToolCall as ProviderToolCall
from taskforge.llm.client import Usage as ProviderUsage
from taskforge.llm.policy import LLMPolicy

from forge.effects.events import Event, Retrying
from forge.effects.issues import Hold, Restart
from forge.model.llm import Completion, Message, Role, ToolCall, ToolSchema
from forge.resources.inference import (
    CompletionBackend,
    ContextExceeded,
    FakeInference,
    GlmInference,
    InferenceRequest,
    complete,
)


class Backend(CompletionBackend):
    model = "test-model"

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


class Retries:
    def __init__(self) -> None:
        self.events: list[Retrying] = []

    def __call__(self, event: Event[None]) -> None:
        assert isinstance(event, Retrying)
        self.events.append(event)


def provider_completion(attempts: tuple[Attempt, ...] = ()) -> ProviderCompletion:
    return ProviderCompletion(
        "answer",
        "reason",
        (ProviderToolCall("id", "shell", "{}"),),
        ProviderFinish.TOOL_CALLS,
        ProviderUsage(10, 20, 3, 4),
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


async def test_inference_retries_then_preserves_wire_and_completion_values() -> None:
    retries = Retries()
    backend = Backend((ModelRequestRejected(503, "private response", ()), provider_completion()))
    event = request()
    result = await complete(event, GlmInference(backend, "provider-config"), notify=retries)
    assert result.content == "answer" and result.reasoning == "reason"
    assert result.tool_calls == (ToolCall("id", "shell", "{}"),)
    assert result.usage.cached_tokens == 4 and result.continuations == 2
    messages, policy, fields = backend.calls[-1]
    assert messages[0]["reasoning_content"] == "thought"
    assert messages[1]["tool_call_id"] == "previous"
    assert policy.max_tokens == 100
    assert fields == {
        "tools": [
            {
                "type": "function",
                "function": {"name": "shell", "description": "Run command", "parameters": {"type": "object"}},
            }
        ]
    }
    assert [(retry.request, retry.attempt) for retry in retries.events] == [(event, 1)]
    assert "private" not in retries.events[0].cause


async def test_inference_exhausts_bound_without_consuming_later_success() -> None:
    retries = Retries()
    backend = Backend(
        (
            ModelRequestRejected(429, "", ()),
            ModelRequestRejected(503, "", ()),
            ModelRequestRejected(503, "", ()),
            provider_completion(),
        )
    )
    event = request()
    with pytest.raises(Restart):
        await complete(event, GlmInference(backend, "provider-config"), notify=retries)
    assert len(backend.calls) == 3
    assert [retry.attempt for retry in retries.events] == [1, 2]


async def test_provider_exhaustion_is_not_retried_by_executor() -> None:
    attempts = (Attempt(0, AttemptOutcome.RETRYABLE_STATUS, 10, 0, 1, 503, "secret", ()),) * 2
    backend = Backend((ModelUnavailable("secret", attempts), provider_completion()))
    retries = Retries()
    event = request()
    with pytest.raises(Restart):
        await complete(event, GlmInference(backend, "provider-config"), notify=retries)
    assert len(backend.calls) == 1
    assert len(retries.events) == 1 and "secret" not in retries.events[0].cause


async def test_request_rejection_holds_without_retry() -> None:
    body = "invalid tool schema: " + "x" * 500 + "excluded suffix"
    backend = Backend((ModelRequestRejected(400, body, ()),))
    retries = Retries()
    with pytest.raises(Hold) as caught:
        await complete(request(), GlmInference(backend, "provider-config"), notify=retries)
    assert caught.value.args[0] == f"GLM request rejected with status 400: {body[:500]}"
    assert len(backend.calls) == 1 and not retries.events


async def test_fake_inference_records_requests_and_stops_at_script_end() -> None:
    client = FakeInference((Completion("first"), Completion("second")))
    event = request()
    assert (await client.complete(event)).content == "first"
    assert (await client.complete(event)).content == "second"
    with pytest.raises(RuntimeError):
        await client.complete(event)
    assert client.requests == [event, event, event]


async def test_executor_publishes_provider_retries_after_success_without_another_call() -> None:
    attempts = (
        Attempt(0, AttemptOutcome.RETRYABLE_STATUS, 10, 0, 1, 503, "private", ()),
        Attempt(0, AttemptOutcome.COMPLETED, 10, 1, 1, 200, "", ()),
    )
    backend = Backend((provider_completion(attempts),))
    retries = Retries()
    event = request()
    result = await complete(event, GlmInference(backend, "provider"), notify=retries)
    assert result.retries == ("retryable_status:503",)
    assert retries.events == [Retrying(event, 1, "retryable_status:503")]
    assert len(backend.calls) == 1


async def test_provider_context_limit_exits_without_retry() -> None:
    body = "This model's maximum context length is 262144 tokens. " + "x" * 500
    rejection = ModelRequestRejected(400, body, ())
    backend = Backend((rejection, provider_completion()))
    retries = Retries()
    with pytest.raises(ContextExceeded) as caught:
        await complete(request(), GlmInference(backend, "provider-config"), notify=retries)
    assert str(caught.value) == body
    assert caught.value.__cause__ is rejection
    assert len(backend.calls) == 1 and not retries.events


async def test_fake_inference_preserves_context_limit_type() -> None:
    error = ContextExceeded("full context")
    with pytest.raises(ContextExceeded) as caught:
        await FakeInference((error,)).complete(request())
    assert caught.value is error
