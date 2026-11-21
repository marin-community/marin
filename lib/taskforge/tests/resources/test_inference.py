# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from lib.taskforge.tests.resources.inference_fakes import Backend, provider_completion, request

from taskforge.effects.events import Event, Retrying
from taskforge.effects.issues import Restart
from taskforge.llm.client import Attempt, AttemptOutcome, ContextExhausted, ModelRequestRejected, ModelUnavailable
from taskforge.model.llm import Completion, FinishReason, ToolCall
from taskforge.resources.inference import (
    GlmInference,
    complete,
)


class Retries:
    def __init__(self) -> None:
        self.events: list[Retrying] = []

    def __call__(self, event: Event[None]) -> None:
        assert isinstance(event, Retrying)
        self.events.append(event)


async def test_retry_telemetry_preserves_wire_and_completion_values() -> None:
    attempts = (
        Attempt(0, AttemptOutcome.RETRYABLE_STATUS, 10, 0, 1, 503, "private", ()),
        Attempt(0, AttemptOutcome.COMPLETED, 10, 1, 1, 200, "", ()),
    )
    backend = Backend((provider_completion(attempts),))
    broken = '{"findings": '
    event = request()
    event = replace(
        event,
        messages=(replace(event.messages[0], tool_calls=(ToolCall("previous", "report", broken),)), event.messages[1]),
    )
    retries = Retries()
    result = await complete(event, GlmInference(backend, "provider"), notify=retries)
    assert (result.content, result.reasoning, result.tool_calls) == (
        "answer",
        "reason",
        (ToolCall("id", "shell", "{}"),),
    )
    assert result.usage.cached_tokens == 4 and result.continuations == 2
    messages, policy, fields = backend.calls[-1]
    assert messages[0]["reasoning_content"] == "thought" and messages[1]["tool_call_id"] == "previous"
    wire = json.loads(json.dumps(messages[0]))
    assert wire["tool_calls"][0]["function"]["arguments"] == broken  # the harness shapes arguments when it records
    assert event.messages[0].tool_calls[0].arguments == broken
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
    assert result.retries == ("retryable_status:503",)
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "error,expected,calls,retry_count",
    [
        (
            ModelUnavailable(
                "private", (Attempt(0, AttemptOutcome.RETRYABLE_STATUS, 10, 0, 1, 503, "private", ()),) * 2
            ),
            Restart,
            1,
            1,
        ),
        *(
            (ModelRequestRejected(status, "malformed request", ()), ModelRequestRejected, 1, 0)
            for status in (400, 401, 403, 404, 422)
        ),
    ],
)
async def test_provider_failures_preserve_classification_and_retry_bound(
    error: ModelRequestRejected | ModelUnavailable, expected: type[Exception], calls: int, retry_count: int
) -> None:
    backend = Backend((error,) * calls + (provider_completion(),))
    retries = Retries()
    with pytest.raises(expected) as caught:
        await complete(request(), GlmInference(backend, "provider"), notify=retries)
    assert len(backend.calls) == calls and len(retries.events) == retry_count
    assert all("private" not in retry.cause for retry in retries.events)
    if expected is ModelRequestRejected:
        assert caught.value is error


async def test_a_context_window_refusal_is_a_recorded_completion_one_request_long() -> None:
    """The provider's context-exhaustion signal is a result, not a raise: the log records it under the request, so a
    resumed request replays the same stop instead of asking the provider again."""
    refusal = ContextExhausted("maximum context length " + "x" * 500, ())
    backend = Backend((refusal, provider_completion()))
    result = await complete(request(), GlmInference(backend, "provider"), notify=Retries())
    assert result == Completion(refusal.body, finish_reason=FinishReason.CONTEXT_EXCEEDED) and len(backend.calls) == 1
