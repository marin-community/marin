# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import socket
from collections.abc import Mapping
from dataclasses import dataclass, replace

import pytest

from taskforge.llm.client import (
    AttemptOutcome,
    Completion,
    ContextExhausted,
    FinishReason,
    ModelRequestRejected,
    ModelUnavailable,
)
from taskforge.llm.glm_client import (
    EndpointSource,
    FixedEndpoint,
    GlmClient,
    GlmEndpoint,
    Pool,
)
from taskforge.llm.policy import (
    CONTINUE_PROMPT,
    GLM_MAX_OUTPUT_TOKENS,
    LLMPolicy,
    LLMProviderConfig,
    Message,
)

MESSAGES: list[Message] = [{"role": "user", "content": "hi"}]
CONTEXT_ERROR = json.dumps(
    {
        "error": {
            "message": (
                "This model's maximum context length is 262144 tokens. However, you requested 131072 "
                "output tokens and your prompt contains at least 131073 input tokens."
            ),
            "code": 400,
        }
    }
)


def endpoint(base_url: str) -> GlmEndpoint:
    return GlmEndpoint(base_url=base_url, token="test-token", pool=Pool.HIGH)


@dataclass
class MovingEndpoint:
    """Answers each ``current`` call with the next endpoint, then keeps answering the last."""

    endpoints: list[GlmEndpoint]
    calls: int = 0

    def current(self) -> GlmEndpoint:
        self.calls += 1
        return self.endpoints[min(self.calls, len(self.endpoints)) - 1]


def complete(
    source: EndpointSource,
    provider: LLMProviderConfig,
    policy: LLMPolicy = LLMPolicy(),
    request_fields: Mapping[str, object] | None = None,
) -> Completion:
    async def call() -> Completion:
        async with GlmClient(source, provider) as client:
            return await client.complete(MESSAGES, policy, request_fields)

    return asyncio.run(call())


@pytest.fixture
def run(fake_glm, fast_provider):
    """Run one ``complete`` of ``MESSAGES`` against ``fake_glm``; keyword arguments override provider fields."""

    def go(policy=LLMPolicy(), **provider_fields) -> Completion:
        return complete(FixedEndpoint(endpoint(fake_glm.base_url)), replace(fast_provider, **provider_fields), policy)

    return go


def outcomes(completion_or_error) -> list[AttemptOutcome]:
    return [a.outcome for a in completion_or_error.attempts]


def test_stream_assembles_content_reasoning_usage_and_chunks(fake_glm, run):
    fake_glm.stream(content="Hello there", reasoning="thinking", prompt_tokens=12, completion_tokens=7)
    completion = run()
    assert (completion.content, completion.reasoning) == ("Hello there", "thinking")
    assert completion.finish_reason is FinishReason.STOP
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (12, 7)
    assert (completion.usage.reasoning_tokens, completion.usage.cached_tokens) == (1, 0)
    [attempt] = completion.attempts
    assert len(attempt.events) == 6
    assert json.loads(attempt.events[1])["choices"][0]["delta"]["reasoning"] == "thin"
    assert json.loads(attempt.events[-1])["usage"]["completion_tokens"] == 7
    request = fake_glm.requests[0]
    assert request["max_tokens"] == GLM_MAX_OUTPUT_TOKENS
    assert (request["stream"], request["stream_options"]) == (True, {"include_usage": True})
    assert request["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_request_fields_reach_the_wire(fake_glm, fast_provider):
    tools = [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}]
    tool_choice = {"type": "function", "function": {"name": "answer"}}
    constraint = {"regex": "[0-9]+"}
    fake_glm.stream(content="42")
    fields = {"tools": tools, "tool_choice": tool_choice, "structured_outputs": constraint}
    complete(FixedEndpoint(endpoint(fake_glm.base_url)), fast_provider, request_fields=fields)
    request = fake_glm.requests[0]
    assert (request["tools"], request["tool_choice"], request["structured_outputs"]) == (tools, tool_choice, constraint)
    assert request["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_tool_call_deltas_are_joined(fake_glm, run):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris"}'),))
    completion = run()
    assert [(c.name, c.arguments) for c in completion.tool_calls] == [("answer", '{"city": "Paris"}')]


@pytest.mark.parametrize(
    ("completion_tokens", "finish"), [(64, FinishReason.LENGTH), (63, FinishReason.TOOL_CALLS)], ids=["at", "under"]
)
def test_tool_call_reply_that_used_its_whole_budget_is_reported_as_length(fake_glm, run, completion_tokens, finish):
    fake_glm.stream(
        tool_calls=(("shell", '{"command": "cat <<EOF'),), finish="tool_calls", completion_tokens=completion_tokens
    )

    completion = run(LLMPolicy(max_tokens=64))

    assert (completion.finish_reason, completion.continuations, len(fake_glm.requests)) == (finish, 0, 1)
    assert json.loads(completion.attempts[0].events[-2])["choices"][0]["finish_reason"] == "tool_calls"


def test_retryable_statuses_are_retried_then_succeed(fake_glm, run):
    fake_glm.status(429, "slow down", {"Retry-After": "0"})
    fake_glm.status(503, "unavailable")
    fake_glm.stream(content="ok")
    completion = run()
    assert completion.content == "ok"
    assert outcomes(completion) == [AttemptOutcome.RETRYABLE_STATUS] * 2 + [AttemptOutcome.COMPLETED]
    assert [a.http_status for a in completion.attempts] == [429, 503, 200]


def test_retry_after_sets_the_minimum_delay(fake_glm, run):
    fake_glm.status(429, "slow down", {"Retry-After": "0.3"})
    fake_glm.stream(content="ok")
    first, second = run().attempts
    assert second.started - (first.started + first.duration) >= 0.3


@pytest.mark.parametrize(
    "stream",
    [
        {"content": "partial", "finish": "abort"},
        {"content": "partial", "raw_payload": "{not json"},
        {"content": "partial", "raw_payload": '{"error": {"message": "engine died", "code": 500}}'},
    ],
    ids=["abort", "malformed", "error-event"],
)
def test_aborted_malformed_or_errored_stream_is_retried(fake_glm, run, stream):
    fake_glm.stream(**stream)
    fake_glm.stream(content="whole")
    completion = run()
    assert completion.content == "whole"
    assert completion.attempts[0].outcome in (AttemptOutcome.ABORTED, AttemptOutcome.STREAM_ERROR)
    assert completion.attempts[1].outcome is AttemptOutcome.COMPLETED


def test_exhausted_attempts_raise_unavailable_with_record(fake_glm, run):
    fake_glm.status(500, "boom")
    fake_glm.status(502, "boom")
    with pytest.raises(ModelUnavailable) as error:
        run(max_attempts=2)
    assert outcomes(error.value) == [AttemptOutcome.RETRYABLE_STATUS] * 2


def test_client_error_is_not_retried(fake_glm, run):
    fake_glm.status(401, "bad token")
    with pytest.raises(ModelRequestRejected) as error:
        run()
    assert error.value.status == 401
    assert len(fake_glm.requests) == 1
    [attempt] = error.value.attempts
    assert (attempt.outcome, attempt.http_status) == (AttemptOutcome.REJECTED, 401)


def test_route_404_holds_without_counting_an_attempt(fake_glm, run):
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    fake_glm.stream(content="back")
    completion = run(max_attempts=1)
    assert completion.content == "back"
    assert outcomes(completion) == [AttemptOutcome.ROUTE_MISSING] * 2 + [AttemptOutcome.COMPLETED]


def test_route_hold_follows_an_endpoint_that_moved(fake_glm, other_fake_glm, fast_provider):
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    other_fake_glm.stream(content="moved")
    source = MovingEndpoint([endpoint(fake_glm.base_url), endpoint(other_fake_glm.base_url)])
    completion = complete(source, replace(fast_provider, max_attempts=1))
    assert completion.content == "moved"
    assert outcomes(completion) == [AttemptOutcome.ROUTE_MISSING, AttemptOutcome.COMPLETED]
    assert (len(fake_glm.requests), len(other_fake_glm.requests)) == (1, 1)


def test_refused_connection_is_retried_on_the_current_endpoint(fake_glm, fast_provider):
    with socket.create_server(("127.0.0.1", 0)) as closed:
        closed_port = closed.getsockname()[1]
    fake_glm.stream(content="reached")
    source = MovingEndpoint([endpoint(f"http://127.0.0.1:{closed_port}/v1"), endpoint(fake_glm.base_url)])
    completion = complete(source, fast_provider)
    assert completion.content == "reached"
    assert outcomes(completion) == [AttemptOutcome.TRANSPORT_ERROR, AttemptOutcome.COMPLETED]


def test_pool_without_workers_holds_until_health_recovers(fake_glm, run):
    fake_glm.health_workers.clear()
    fake_glm.health_workers.extend([{"high": 0}, {"high": 0}, {"high": 0}, {"high": 2}])
    fake_glm.status(502, "upstream gone")
    fake_glm.stream(content="recovered")
    completion = run(max_attempts=1)
    assert completion.content == "recovered"
    assert fake_glm.health_polls == 4


def test_hold_gives_up_after_hold_timeout(fake_glm, run):
    fake_glm.health_workers.clear()
    fake_glm.health_workers.append({"high": 0, "bulk": 5})
    fake_glm.status(503, "no workers")
    with pytest.raises(ModelUnavailable, match="hold"):
        run(hold_timeout=0.05)


@pytest.mark.parametrize(
    ("health_status", "workers"),
    [(500, {"high": 0}), (200, {"bulk": 0})],
    ids=["health-500", "pool-not-reported"],
)
def test_failure_counts_an_attempt_unless_health_reports_the_pool_empty(fake_glm, run, health_status, workers):
    fake_glm.health_status = health_status
    fake_glm.health_workers.clear()
    fake_glm.health_workers.append(workers)
    fake_glm.status(503, "busy")
    with pytest.raises(ModelUnavailable, match="attempts exhausted"):
        run(max_attempts=1, hold_timeout=60.0)


def test_stall_abandons_attempt_and_retries(fake_glm, run):
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")
    completion = run(stall_timeout=0.3)
    assert completion.content == "second try"
    stalled = completion.attempts[0]
    assert stalled.outcome is AttemptOutcome.STALLED
    assert len(stalled.events) == 2


def test_stream_cut_before_finish_reason_is_retried(fake_glm, run):
    fake_glm.stream(content="cut", finish=None, send_done=False)
    fake_glm.stream(content="whole")
    completion = run()
    assert completion.content == "whole"
    assert outcomes(completion) == [AttemptOutcome.INCOMPLETE_STREAM, AttemptOutcome.COMPLETED]


def test_context_overflow_raises_without_lowering_max_tokens(fake_glm, run):
    fake_glm.status(400, CONTEXT_ERROR)
    with pytest.raises(ContextExhausted) as error:
        run()
    assert error.value.status == 400
    assert error.value.attempts[0].outcome is AttemptOutcome.CONTEXT_OVERFLOW
    assert [r["max_tokens"] for r in fake_glm.requests] == [GLM_MAX_OUTPUT_TOKENS]


def test_length_finish_continues_from_partial_output(fake_glm, run):
    fake_glm.stream(content="first half, ", finish="length", completion_tokens=100)
    fake_glm.stream(content="second half.", finish="stop", completion_tokens=40)
    completion = run()
    assert completion.content == "first half, second half."
    assert completion.continuations == 1
    assert completion.usage.completion_tokens == 140
    assert fake_glm.requests[1]["messages"] == [
        *MESSAGES,
        {"role": "assistant", "content": "first half, "},
        {"role": "user", "content": CONTINUE_PROMPT},
    ]


def test_reply_cut_off_while_reasoning_extends_the_open_reasoning_turn(fake_glm, run):
    fake_glm.stream(reasoning="step one, ", finish="length")
    fake_glm.stream(reasoning="step two, ", finish="length")
    fake_glm.stream(reasoning="step three", content="answer, cut", finish="length")
    fake_glm.stream(content=" off")
    completion = run()
    assert (completion.reasoning, completion.content) == ("step one, step two, step three", "answer, cut off")
    second, third, fourth = fake_glm.requests[1:]
    assert second["messages"] == [*MESSAGES, {"role": "assistant", "content": "<think>step one, "}]
    assert (second["continue_final_message"], second["add_generation_prompt"]) == (True, False)
    assert third["messages"][-1] == {"role": "assistant", "content": "<think>step one, step two, "}
    assert fourth["messages"] == [
        *MESSAGES,
        {"role": "assistant", "content": "answer, cut"},
        {"role": "user", "content": CONTINUE_PROMPT},
    ]
    assert "continue_final_message" not in fourth


def test_length_with_a_tool_call_is_not_continued(fake_glm, run):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Par'),), finish="length")
    completion = run()
    assert completion.finish_reason is FinishReason.LENGTH
    assert completion.continuations == 0
    assert len(fake_glm.requests) == 1


def test_continuation_that_no_longer_fits_returns_the_partial_output(fake_glm, run):
    fake_glm.stream(content="long answer", finish="length", prompt_tokens=140_020, completion_tokens=122_124)
    fake_glm.status(400, CONTEXT_ERROR)
    completion = run()
    assert completion.content == "long answer"
    assert completion.finish_reason is FinishReason.LENGTH
    assert completion.continuations == 0
    assert [(a.segment, a.outcome) for a in completion.attempts] == [
        (0, AttemptOutcome.COMPLETED),
        (1, AttemptOutcome.CONTEXT_OVERFLOW),
    ]


def test_continuations_stop_at_policy_limit(fake_glm, run):
    for part in ("a", "b", "c"):
        fake_glm.stream(content=part, finish="length")
    completion = run(policy=LLMPolicy(max_continuations=2))
    assert completion.content == "abc"
    assert completion.finish_reason is FinishReason.LENGTH
    assert len(fake_glm.requests) == 3


def test_a_stalled_read_retries_through_a_fresh_connection_pool(fake_glm, fast_provider):
    """Ladder harness patch: a silently dead keep-alive socket reads nothing until the stall timeout, and so would
    every other idle connection the pool holds on that path, so the retry goes through a new pool. (The fake server
    speaks HTTP/1.0 and never keeps a connection alive, so the test checks the pool, not the socket.)"""
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")

    async def call() -> tuple[Completion, bool]:
        provider = replace(fast_provider, stall_timeout=0.3)
        async with GlmClient(FixedEndpoint(endpoint(fake_glm.base_url)), provider) as client:
            before = client.http
            completion = await client.complete(MESSAGES, LLMPolicy())
            return completion, client.http is not before and before.is_closed

    completion, replaced_and_closed = asyncio.run(call())
    assert completion.content == "second try"
    assert outcomes(completion) == [AttemptOutcome.STALLED, AttemptOutcome.COMPLETED]
    assert replaced_and_closed


def test_a_replaced_pool_closes_once_the_calls_still_streaming_on_it_finish(fake_glm, fast_provider):
    """A pool replaced after a stall must not hold its sockets until the client exits, nor cut a call streaming on it."""
    fake_glm.stream(reasoning="slow but steady", content="first", interval=0.2)
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")

    async def call() -> tuple[str, bool, bool, Completion]:
        provider = replace(fast_provider, stall_timeout=0.4)
        async with GlmClient(FixedEndpoint(endpoint(fake_glm.base_url)), provider) as client:
            before = client.http
            streaming = asyncio.create_task(client.complete(MESSAGES, LLMPolicy()))
            while not fake_glm.requests:
                await asyncio.sleep(0.01)
            retried = await client.complete(MESSAGES, LLMPolicy())
            open_while_streaming = not streaming.done() and not before.is_closed
            slow = await streaming
            return retried.content, open_while_streaming, before.is_closed, slow

    retried, open_while_streaming, closed_after, slow = asyncio.run(call())
    assert retried == "second try"
    assert open_while_streaming
    assert (slow.content, outcomes(slow)) == ("first", [AttemptOutcome.COMPLETED])
    assert closed_after
