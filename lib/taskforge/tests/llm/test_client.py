# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import pickle
import socket
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from lib.taskforge.tests.llm.fake_glm import GLM_MODEL, FakeGlm
from lib.taskforge.tests.resources.inference_fakes import request

from taskforge import wiring
from taskforge.effects.issues import Restart
from taskforge.llm import glm_provider
from taskforge.llm.client import (
    AttemptOutcome,
    Completion,
    ContextExhausted,
    ModelClient,
    ModelRequestRejected,
    ModelUnavailable,
)
from taskforge.llm.glm_provider import (
    EndpointSource,
    EndpointUnavailable,
    FixedEndpoint,
    GlmEndpoint,
    GlmProvider,
    Pool,
)
from taskforge.llm.policy import (
    CONTINUE_PROMPT,
    GLM_5_3,
    GLM_MAX_OUTPUT_TOKENS,
    LLMPolicy,
    Message,
    RetryPolicy,
)
from taskforge.model.llm import FinishReason
from taskforge.resources.inference import GlmInference

MESSAGES: list[Message] = [{"role": "user", "content": "hi"}]
LIMITED = replace(GLM_5_3, max_output_tokens=GLM_MAX_OUTPUT_TOKENS)
"""A profile that limits output, so requests carry ``max_tokens``; the GLM-5.3 default leaves it to the endpoint."""
Run = Callable[..., Completion]
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
    """Answers each ``current`` call with the next endpoint, then keeps answering the last; None is no endpoint."""

    endpoints: list[GlmEndpoint | None]
    calls: int = 0

    def current(self) -> GlmEndpoint:
        self.calls += 1
        answer = self.endpoints[min(self.calls, len(self.endpoints)) - 1]
        if answer is None:
            raise EndpointUnavailable("relay not registered")
        return answer


def provider(source: EndpointSource, stall_timeout: float = 600.0) -> GlmProvider:
    return GlmProvider(source, GLM_MODEL, stall_timeout, 8)


def complete(
    source: EndpointSource,
    retry: RetryPolicy,
    policy: LLMPolicy = LLMPolicy(),
    request_fields: Mapping[str, object] | None = None,
    stall_timeout: float = 600.0,
) -> Completion:
    async def call() -> Completion:
        async with provider(source, stall_timeout) as glm:
            return await ModelClient(glm, LIMITED, retry, role="test").complete(MESSAGES, policy, request_fields)

    return asyncio.run(call())


@pytest.fixture
def run(fake_glm: FakeGlm, fast_retry: RetryPolicy) -> Run:
    """Run one ``complete`` of ``MESSAGES`` against ``fake_glm``."""

    def go(policy: LLMPolicy = LLMPolicy(), stall_timeout: float = 600.0) -> Completion:
        return complete(FixedEndpoint(endpoint(fake_glm.base_url)), fast_retry, policy, stall_timeout=stall_timeout)

    return go


def outcomes(completion_or_error: Completion | ModelUnavailable) -> list[AttemptOutcome]:
    return [a.outcome for a in completion_or_error.attempts]


def test_stream_assembles_content_reasoning_usage_and_chunks(fake_glm: FakeGlm, run: Run) -> None:
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


def test_request_fields_reach_the_wire(fake_glm: FakeGlm, fast_retry: RetryPolicy) -> None:
    tools = [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}]
    tool_choice = {"type": "function", "function": {"name": "answer"}}
    constraint = {"regex": "[0-9]+"}
    fake_glm.stream(content="42")
    fields = {"tools": tools, "tool_choice": tool_choice, "structured_outputs": constraint}
    complete(FixedEndpoint(endpoint(fake_glm.base_url)), fast_retry, request_fields=fields)
    request = fake_glm.requests[0]
    assert (request["tools"], request["tool_choice"], request["structured_outputs"]) == (tools, tool_choice, constraint)
    assert request["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_tool_call_deltas_are_joined(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris"}'),))
    completion = run()
    assert [(c.name, c.arguments) for c in completion.tool_calls] == [("answer", '{"city": "Paris"}')]


@pytest.mark.parametrize(
    ("completion_tokens", "finish"), [(64, FinishReason.LENGTH), (63, FinishReason.TOOL_CALLS)], ids=["at", "under"]
)
def test_tool_call_reply_that_used_its_whole_budget_is_reported_as_length(
    fake_glm: FakeGlm, run: Run, completion_tokens: int, finish: FinishReason
) -> None:
    fake_glm.stream(
        tool_calls=(("shell", '{"command": "cat <<EOF'),), finish="tool_calls", completion_tokens=completion_tokens
    )

    completion = run(LLMPolicy(max_tokens=64))

    assert (completion.finish_reason, completion.continuations, len(fake_glm.requests)) == (finish, 0, 1)
    assert json.loads(completion.attempts[0].events[-2])["choices"][0]["finish_reason"] == "tool_calls"


def test_retryable_statuses_are_retried_then_succeed(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.status(429, "slow down", {"Retry-After": "0"})
    fake_glm.status(503, "unavailable")
    fake_glm.stream(content="ok")
    completion = run()
    assert completion.content == "ok"
    assert outcomes(completion) == [AttemptOutcome.RETRYABLE_STATUS] * 2 + [AttemptOutcome.COMPLETED]
    assert [a.http_status for a in completion.attempts] == [429, 503, 200]


def test_retry_after_sets_the_minimum_delay(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.status(429, "slow down", {"Retry-After": "0.3"})
    fake_glm.stream(content="ok")
    first, second = run().attempts
    assert second.started - (first.started + first.duration) >= 0.3


# A tool-call delta for index 1 with no earlier delta, which the SDK accumulator cannot place.
ORPHAN_TOOL_DELTA = json.dumps(
    {
        "choices": [
            {"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "x"}}]}, "finish_reason": None}
        ]
    }
)


@pytest.mark.parametrize(
    "stream",
    [
        {"content": "partial", "finish": "abort"},
        {"content": "partial", "raw_payload": "{not json"},
        {"content": "partial", "raw_payload": '{"error": {"message": "engine died", "code": 500}}'},
        {"content": "partial", "raw_payload": ORPHAN_TOOL_DELTA},
    ],
    ids=["abort", "malformed", "error-event", "orphan-tool-delta"],
)
def test_aborted_malformed_or_errored_stream_is_retried(fake_glm: FakeGlm, run: Run, stream: dict[str, str]) -> None:
    fake_glm.stream(
        content=stream["content"], finish=stream.get("finish", "stop"), raw_payload=stream.get("raw_payload")
    )
    fake_glm.stream(content="whole")
    completion = run()
    assert completion.content == "whole"
    assert completion.attempts[0].outcome in (AttemptOutcome.ABORTED, AttemptOutcome.STREAM_ERROR)
    assert completion.attempts[1].outcome is AttemptOutcome.COMPLETED


def test_client_error_is_not_retried(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.status(401, "bad token")
    with pytest.raises(ModelRequestRejected) as error:
        run()
    assert error.value.status == 401
    assert len(fake_glm.requests) == 1
    [attempt] = error.value.attempts
    assert (attempt.outcome, attempt.http_status) == (AttemptOutcome.REJECTED, 401)


def test_route_404_is_waited_out(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    fake_glm.stream(content="back")
    completion = run()
    assert completion.content == "back"
    assert outcomes(completion) == [AttemptOutcome.ROUTE_MISSING] * 2 + [AttemptOutcome.COMPLETED]


def test_route_hold_follows_an_endpoint_that_moved(
    fake_glm: FakeGlm, other_fake_glm: FakeGlm, fast_retry: RetryPolicy
) -> None:
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    other_fake_glm.stream(content="moved")
    source = MovingEndpoint([endpoint(fake_glm.base_url), endpoint(other_fake_glm.base_url)])
    completion = complete(source, fast_retry)
    assert completion.content == "moved"
    assert outcomes(completion) == [AttemptOutcome.ROUTE_MISSING, AttemptOutcome.COMPLETED]
    assert (len(fake_glm.requests), len(other_fake_glm.requests)) == (1, 1)


def test_refused_connection_is_retried_on_the_current_endpoint(fake_glm: FakeGlm, fast_retry: RetryPolicy) -> None:
    with socket.create_server(("127.0.0.1", 0)) as closed:
        closed_port = closed.getsockname()[1]
    fake_glm.stream(content="reached")
    source = MovingEndpoint([endpoint(f"http://127.0.0.1:{closed_port}/v1"), endpoint(fake_glm.base_url)])
    completion = complete(source, fast_retry)
    assert completion.content == "reached"
    assert outcomes(completion) == [AttemptOutcome.TRANSPORT_ERROR, AttemptOutcome.COMPLETED]


def test_an_unregistered_relay_resolves_at_the_first_request_and_is_waited_out(
    fake_glm: FakeGlm, fast_retry: RetryPolicy
) -> None:
    fake_glm.stream(content="registered")
    source = MovingEndpoint([None, endpoint(fake_glm.base_url)])
    provider(source)
    assert source.calls == 0
    completion = complete(source, fast_retry)
    assert completion.content == "registered"
    assert outcomes(completion) == [AttemptOutcome.TRANSPORT_ERROR, AttemptOutcome.COMPLETED]
    assert len(fake_glm.requests) == 1


def test_pool_without_workers_holds_until_health_recovers(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.health_workers.clear()
    fake_glm.health_workers.extend([{"high": 0}, {"high": 0}, {"high": 0}, {"high": 2}])
    fake_glm.status(502, "upstream gone")
    fake_glm.stream(content="recovered")
    completion = run()
    assert completion.content == "recovered"
    assert fake_glm.health_polls == 4
    assert len(fake_glm.requests) == 2  # the hold polls /health instead of resending


@pytest.mark.parametrize(
    ("health_status", "workers"),
    [(500, {"high": 0}), (200, {"bulk": 0})],
    ids=["health-500", "pool-not-reported"],
)
def test_a_failure_is_resent_after_one_backoff_unless_health_reports_the_pool_empty(
    fake_glm: FakeGlm, run: Run, health_status: int, workers: dict[str, int]
) -> None:
    fake_glm.health_status = health_status
    fake_glm.health_workers.clear()
    fake_glm.health_workers.append(workers)
    fake_glm.status(503, "busy")
    fake_glm.stream(content="served")
    assert run().content == "served"
    assert fake_glm.health_polls == 1


def test_stall_abandons_attempt_and_retries(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")
    completion = run(stall_timeout=0.3)
    assert completion.content == "second try"
    stalled = completion.attempts[0]
    assert stalled.outcome is AttemptOutcome.STALLED
    assert len(stalled.events) == 2


def test_stream_cut_before_finish_reason_is_retried(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.stream(content="cut", finish=None, send_done=False)
    fake_glm.stream(content="whole")
    completion = run()
    assert completion.content == "whole"
    assert outcomes(completion) == [AttemptOutcome.INCOMPLETE_STREAM, AttemptOutcome.COMPLETED]


@pytest.mark.parametrize(
    "body",
    [
        CONTEXT_ERROR,
        json.dumps({"error": {"message": "Input length (300000) exceeds model's maximum context length (262144)."}}),
        json.dumps({"error": {"message": "prompt is longer than the maximum model length of 262144"}}),
        json.dumps({"error": {"message": "too many tokens", "code": "context_length_exceeded"}}),
    ],
    ids=["vllm-requested", "vllm-input-length", "vllm-engine", "openai-code"],
)
def test_context_overflow_raises_without_lowering_max_tokens(fake_glm: FakeGlm, run: Run, body: str) -> None:
    fake_glm.status(400, body)
    with pytest.raises(ContextExhausted) as error:
        run()
    assert error.value.status == 400
    assert error.value.attempts[0].outcome is AttemptOutcome.CONTEXT_OVERFLOW
    assert [r["max_tokens"] for r in fake_glm.requests] == [GLM_MAX_OUTPUT_TOKENS]


def test_length_finish_continues_from_partial_output(fake_glm: FakeGlm, run: Run) -> None:
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


def test_reply_cut_off_while_reasoning_extends_the_open_reasoning_turn(fake_glm: FakeGlm, run: Run) -> None:
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


def test_length_with_a_tool_call_is_not_continued(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.stream(tool_calls=(("answer", '{"city": "Par'),), finish="length")
    completion = run()
    assert completion.finish_reason is FinishReason.LENGTH
    assert completion.continuations == 0
    assert len(fake_glm.requests) == 1


def test_continuation_that_no_longer_fits_returns_the_partial_output(fake_glm: FakeGlm, run: Run) -> None:
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


def test_continuations_stop_at_policy_limit(fake_glm: FakeGlm, run: Run) -> None:
    for part in ("a", "b", "c"):
        fake_glm.stream(content=part, finish="length")
    completion = run(policy=LLMPolicy(max_continuations=2))
    assert completion.content == "abc"
    assert completion.finish_reason is FinishReason.LENGTH
    assert len(fake_glm.requests) == 3


def test_a_stalled_read_retries_through_a_fresh_connection_pool(fake_glm: FakeGlm, fast_retry: RetryPolicy) -> None:
    """A silently dead keep-alive socket reads nothing until the stall timeout, and so would
    every other idle connection the pool holds on that path, so the retry goes through a new pool. (The fake server
    speaks HTTP/1.0 and never keeps a connection alive, so the test checks the pool, not the socket.)"""
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")

    async def call() -> tuple[Completion, bool]:
        async with provider(FixedEndpoint(endpoint(fake_glm.base_url)), 0.3) as glm:
            before = glm.http
            completion = await ModelClient(glm, LIMITED, fast_retry, role="test").complete(MESSAGES, LLMPolicy())
            return completion, glm.http is not before and before.is_closed

    completion, replaced_and_closed = asyncio.run(call())
    assert completion.content == "second try"
    assert outcomes(completion) == [AttemptOutcome.STALLED, AttemptOutcome.COMPLETED]
    assert replaced_and_closed


def test_a_replaced_pool_closes_once_the_calls_still_streaming_on_it_finish(
    fake_glm: FakeGlm, fast_retry: RetryPolicy
) -> None:
    """A pool replaced after a stall must not hold its sockets until the client exits, nor cut a call streaming on it."""
    fake_glm.stream(reasoning="slow but steady", content="first", interval=0.2)
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")

    async def call() -> tuple[str, bool, bool, Completion]:
        async with provider(FixedEndpoint(endpoint(fake_glm.base_url)), 0.4) as glm:
            client = ModelClient(glm, LIMITED, fast_retry, role="test")
            before = glm.http
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


def test_a_pickled_client_completes_in_each_new_event_loop(fake_glm: FakeGlm, fast_retry: RetryPolicy) -> None:
    fake_glm.stream(content="first")
    fake_glm.stream(content="second")
    glm = provider(FixedEndpoint(endpoint(fake_glm.base_url)))
    client = pickle.loads(pickle.dumps(ModelClient(glm, LIMITED, fast_retry, role="test")))

    # Each asyncio.run is a new event loop.
    replies = [asyncio.run(client.complete(MESSAGES, LLMPolicy())).content for _ in range(2)]

    assert replies == ["first", "second"]


class KeepAliveHealth(BaseHTTPRequestHandler):
    """Answers ``/health`` and keeps the connection open, as the router does."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (the base class names it format)
        pass


def test_the_connection_pool_of_one_client_works_in_each_new_event_loop() -> None:
    with ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveHealth) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        glm = provider(FixedEndpoint(endpoint(f"{base_url}/v1")))

        async def health() -> int:
            return (await glm.http.get(f"{base_url}/health")).status_code

        # A pool that a closed loop opened fails on its kept-alive connection with "Event loop is closed".
        assert [asyncio.run(health()) for _ in range(2)] == [200, 200]
        server.shutdown()


def test_endpoint_output_limit_survives_pickling_and_preserves_tool_calls(
    fake_glm: FakeGlm, fast_retry: RetryPolicy
) -> None:
    """The GLM-5.3 profile leaves output to the endpoint: no token-limit field, no truncation heuristic."""
    fake_glm.stream(tool_calls=(("answer", '{"value": 1}'),), finish="tool_calls", completion_tokens=200000)
    glm = pickle.loads(pickle.dumps(provider(FixedEndpoint(endpoint(fake_glm.base_url)))))

    async def call() -> Completion:
        async with glm:
            return await ModelClient(glm, GLM_5_3, fast_retry, role="test").complete(
                MESSAGES, LLMPolicy(max_tokens=1), {"max_tokens": 2, "max_completion_tokens": 3}
            )

    result = asyncio.run(call())
    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert len(fake_glm.requests) == 1
    assert "max_tokens" not in fake_glm.requests[0] and "max_completion_tokens" not in fake_glm.requests[0]
    assert result.attempts[0].max_tokens == 0


# Review 2026-11-21: with the only pooled connection leased, a solver call made no request and ended at `length`.
def test_a_call_waiting_for_a_local_connection_is_unavailable_not_a_generation(
    fake_glm: FakeGlm, fast_retry: RetryPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(glm_provider, "POOL_TIMEOUT", 0.05, raising=False)
    fake_glm.stream(content="holds the only connection", stall_after_first=True)
    bounded = RetryPolicy(0.3, 0.01, 0.01, 1.0, generation_timeout=0.2)

    async def call() -> ModelUnavailable:
        async with GlmProvider(FixedEndpoint(endpoint(fake_glm.base_url)), GLM_MODEL, 600.0, 1) as glm:
            holder = asyncio.create_task(
                ModelClient(glm, LIMITED, fast_retry, role="author").complete(MESSAGES, LLMPolicy())
            )
            while not fake_glm.requests:
                await asyncio.sleep(0.01)
            with pytest.raises(ModelUnavailable) as raised:
                await ModelClient(glm, LIMITED, bounded, role="solver").complete(MESSAGES, LLMPolicy())
            holder.cancel()
            return raised.value

    error = asyncio.run(call())
    assert len(fake_glm.requests) == 1
    assert set(outcomes(error)) == {AttemptOutcome.TRANSPORT_ERROR}


# Review 2026-11-21: a 529 was a rejection; a 503 then an HTML /health reply escaped as JSONDecodeError.
@pytest.mark.parametrize(("status", "health"), [(529, None), (503, "<html>gateway</html>")], ids=["529", "html-health"])
def test_every_5xx_is_an_outage_and_an_undecodable_health_reply_is_unknown(
    fake_glm: FakeGlm, run: Run, status: int, health: str | None
) -> None:
    fake_glm.health_text = health
    fake_glm.status(status, "overloaded")
    fake_glm.stream(content="served")
    completion = run()
    assert completion.content == "served"
    assert outcomes(completion) == [AttemptOutcome.RETRYABLE_STATUS, AttemptOutcome.COMPLETED]


def test_a_cut_in_a_continuation_resends_only_that_segment(fake_glm: FakeGlm, run: Run) -> None:
    fake_glm.stream(content="A", finish="length", completion_tokens=1)
    fake_glm.stream(content="cut", finish=None, send_done=False)
    fake_glm.stream(content="Z", completion_tokens=1)
    completion = run()
    assert (completion.content, completion.continuations, completion.usage.completion_tokens) == ("AZ", 1, 2)
    assert [(a.segment, a.outcome) for a in completion.attempts] == [
        (0, AttemptOutcome.COMPLETED),
        (1, AttemptOutcome.INCOMPLETE_STREAM),
        (1, AttemptOutcome.COMPLETED),
    ]
    assert fake_glm.requests[1]["messages"] == fake_glm.requests[2]["messages"]


@dataclass
class BlockingEndpoint:
    """A registry lookup that blocks its thread until ``released``, as an Iris lookup retrying a dead controller does."""

    released: threading.Event
    calls: int = 0

    def current(self) -> GlmEndpoint:
        self.calls += 1
        self.released.wait()
        raise EndpointUnavailable("released")


# Review 2026-11-21: the synchronous registry lookup, with its 1,800 s retry budget, ran on the event loop.
def test_a_blocked_registry_lookup_is_bounded_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glm_provider, "RESOLVE_TIMEOUT", 0.05, raising=False)
    source = BlockingEndpoint(threading.Event())
    retry = RetryPolicy(0.3, 0.01, 0.01, 1.0)

    async def call() -> int:
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.create_task(tick())
        try:
            async with provider(source) as glm:
                with pytest.raises(ModelUnavailable):
                    await ModelClient(glm, LIMITED, retry, role="author").complete(MESSAGES, LLMPolicy())
        finally:
            ticker.cancel()
            source.released.set()
        return ticks

    assert asyncio.run(call()) >= 10
    assert source.calls == 1


class FailingRegistry:
    """An Iris client whose controller answers every endpoint lookup with 503."""

    def resolver_for_job(self, job: object) -> "FailingRegistry":
        return self

    def resolve(self, name: str) -> object:
        raise ConnectError(Code.UNAVAILABLE, "controller returned 503")


# Review 2026-11-21: a controller 503 during the lazy relay lookup escaped author inference as a raw ConnectError.
def test_a_registry_outage_during_relay_lookup_is_an_unavailable_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wiring, "iris_ctx", lambda: SimpleNamespace(client=FailingRegistry()))
    source = wiring.RelayEndpoint("/me/glm53-relay", "token", Pool.HIGH, GLM_MODEL)
    retry = RetryPolicy(0.05, 0.01, 0.01, 1.0)

    async def call() -> None:
        async with provider(source) as glm:
            await GlmInference(ModelClient(glm, LIMITED, retry, role="author"), "author").complete(request())

    with pytest.raises(Restart, match="unavailable"):
        asyncio.run(call())
