# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import (
    AttemptOutcome,
    Completion,
    FinishReason,
    GlmClient,
    GlmContextExhausted,
    GlmEndpoint,
    GlmRequestRejected,
    GlmUnavailable,
    Pool,
)
from taskforge.llm.policy import CONTINUE_PROMPT, GLM_MAX_OUTPUT_TOKENS, LLMPolicy

MESSAGES = [{"role": "user", "content": "hi"}]
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


def run(fake_glm, policy=LLMPolicy(), max_attempts=8, hold_timeout=60.0) -> Completion:
    async def go() -> Completion:
        endpoint = GlmEndpoint(base_url=fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        backoff = ExponentialBackoff(initial=0.001, maximum=0.001)
        async with GlmClient(endpoint, max_attempts=max_attempts, hold_timeout=hold_timeout, backoff=backoff) as client:
            return await client.complete(MESSAGES, policy)

    return asyncio.run(go())


def outcomes(completion_or_error) -> list[AttemptOutcome]:
    return [a.outcome for a in completion_or_error.attempts]


def test_stream_assembles_content_reasoning_usage_and_raw_events(fake_glm):
    fake_glm.stream(content="Hello there", reasoning="thinking", prompt_tokens=12, completion_tokens=7)
    completion = run(fake_glm)
    assert (completion.content, completion.reasoning) == ("Hello there", "thinking")
    assert completion.finish_reason is FinishReason.STOP
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (12, 7)
    assert json.loads(completion.attempts[0].events[-1])["usage"]["completion_tokens"] == 7
    request = fake_glm.requests[0]
    assert request["max_tokens"] == GLM_MAX_OUTPUT_TOKENS
    assert request["stream"] is True
    assert request["chat_template_kwargs"] == {"reasoning_effort": "high"}


def test_tool_call_deltas_are_joined(fake_glm):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Paris"}'),))
    completion = run(fake_glm)
    assert [(c.name, c.arguments) for c in completion.tool_calls] == [("answer", '{"city": "Paris"}')]


@pytest.mark.parametrize(
    ("completion_tokens", "finish"), [(64, FinishReason.LENGTH), (63, FinishReason.TOOL_CALLS)], ids=["at", "under"]
)
def test_tool_call_reply_that_spent_its_whole_budget_is_reported_as_length(fake_glm, completion_tokens, finish):
    fake_glm.stream(
        tool_calls=(("shell", '{"command": "cat <<EOF'),), finish="tool_calls", completion_tokens=completion_tokens
    )

    completion = run(fake_glm, LLMPolicy(max_tokens=64))

    assert (completion.finish_reason, completion.continuations, len(fake_glm.requests)) == (finish, 0, 1)
    assert json.loads(completion.attempts[0].events[-2])["choices"][0]["finish_reason"] == "tool_calls"


def test_retryable_statuses_spend_attempts_then_succeed(fake_glm):
    fake_glm.status(429, "slow down", {"Retry-After": "0"})
    fake_glm.status(503, "unavailable")
    fake_glm.stream(content="ok")
    completion = run(fake_glm)
    assert completion.content == "ok"
    assert outcomes(completion) == [AttemptOutcome.RETRYABLE_STATUS] * 2 + [AttemptOutcome.COMPLETED]
    assert [a.http_status for a in completion.attempts] == [429, 503, 200]


def test_retry_after_sets_the_minimum_delay(fake_glm):
    fake_glm.status(429, "slow down", {"Retry-After": "0.3"})
    fake_glm.stream(content="ok")
    first, second = run(fake_glm).attempts
    assert second.started - (first.started + first.duration) >= 0.3


@pytest.mark.parametrize(
    "stream",
    [
        {"content": "partial", "finish": "abort"},
        {"content": "partial", "raw_payload": "{not json"},
    ],
    ids=["abort", "malformed"],
)
def test_aborted_or_malformed_stream_is_retried(fake_glm, stream):
    fake_glm.stream(**stream)
    fake_glm.stream(content="whole")
    completion = run(fake_glm)
    assert completion.content == "whole"
    assert completion.attempts[0].outcome in (AttemptOutcome.ABORTED, AttemptOutcome.STREAM_ERROR)
    assert completion.attempts[1].outcome is AttemptOutcome.COMPLETED


def test_exhausted_attempts_raise_unavailable_with_record(fake_glm):
    fake_glm.status(500, "boom")
    fake_glm.status(502, "boom")
    with pytest.raises(GlmUnavailable) as error:
        run(fake_glm, max_attempts=2)
    assert outcomes(error.value) == [AttemptOutcome.RETRYABLE_STATUS] * 2


def test_client_error_is_not_retried(fake_glm):
    fake_glm.status(401, "bad token")
    with pytest.raises(GlmRequestRejected) as error:
        run(fake_glm)
    assert error.value.status == 401
    assert len(fake_glm.requests) == 1


def test_route_404_holds_without_spending_an_attempt(fake_glm):
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    fake_glm.status(404, '{"error": "no route for model glm-5.3"}')
    fake_glm.stream(content="back")
    completion = run(fake_glm, max_attempts=1)
    assert completion.content == "back"
    assert outcomes(completion) == [AttemptOutcome.ROUTE_MISSING] * 2 + [AttemptOutcome.COMPLETED]


def test_pool_without_workers_holds_until_health_recovers(fake_glm):
    fake_glm.health_workers.clear()
    fake_glm.health_workers.extend([{"high": 0}, {"high": 0}, {"high": 0}, {"high": 2}])
    fake_glm.status(502, "upstream gone")
    fake_glm.stream(content="recovered")
    completion = run(fake_glm, max_attempts=1)
    assert completion.content == "recovered"
    assert fake_glm.health_polls == 4


def test_hold_gives_up_after_hold_timeout(fake_glm):
    fake_glm.health_workers.clear()
    fake_glm.health_workers.append({"high": 0, "bulk": 5})
    fake_glm.status(503, "no workers")
    with pytest.raises(GlmUnavailable, match="hold"):
        run(fake_glm, hold_timeout=0.05)


@pytest.mark.parametrize(
    ("health_status", "workers"),
    [(500, {"high": 0}), (200, {"bulk": 0})],
    ids=["health-500", "pool-not-reported"],
)
def test_failure_spends_an_attempt_unless_health_reports_the_pool_empty(fake_glm, health_status, workers):
    fake_glm.health_status = health_status
    fake_glm.health_workers.clear()
    fake_glm.health_workers.append(workers)
    fake_glm.status(503, "busy")
    with pytest.raises(GlmUnavailable, match="attempts exhausted"):
        run(fake_glm, max_attempts=1, hold_timeout=60.0)


def test_stall_abandons_attempt_and_retries(fake_glm):
    fake_glm.stream(content="never finishes", stall_after_first=True)
    fake_glm.stream(content="second try")
    completion = run(fake_glm, policy=LLMPolicy(stall_timeout=0.3))
    assert completion.content == "second try"
    stalled = completion.attempts[0]
    assert stalled.outcome is AttemptOutcome.STALLED
    assert len(stalled.events) == 2


def test_stream_without_done_is_retried(fake_glm):
    fake_glm.stream(content="cut", send_done=False)
    fake_glm.stream(content="whole")
    completion = run(fake_glm)
    assert completion.content == "whole"
    assert outcomes(completion) == [AttemptOutcome.INCOMPLETE_STREAM, AttemptOutcome.COMPLETED]


def test_context_overflow_lowers_max_tokens_to_remaining_context(fake_glm):
    fake_glm.status(400, CONTEXT_ERROR)
    fake_glm.stream(content="x", finish="length", prompt_tokens=140_000, completion_tokens=1)
    fake_glm.stream(content="full answer")
    completion = run(fake_glm)
    assert completion.content == "full answer"
    assert [r["max_tokens"] for r in fake_glm.requests] == [GLM_MAX_OUTPUT_TOKENS, 1, 262_144 - 140_000]
    assert [a.segment for a in completion.attempts] == [0, -1, 0]


def test_context_overflow_that_the_lowered_budget_does_not_fix_is_rejected(fake_glm):
    for _ in range(2):
        fake_glm.status(400, CONTEXT_ERROR)
        fake_glm.stream(content="x", finish="length", prompt_tokens=140_000, completion_tokens=1)
    with pytest.raises(GlmContextExhausted):
        run(fake_glm)
    remaining = 262_144 - 140_000
    assert [r["max_tokens"] for r in fake_glm.requests] == [GLM_MAX_OUTPUT_TOKENS, 1, remaining, 1]


def test_length_finish_continues_from_partial_output(fake_glm):
    fake_glm.stream(content="first half, ", finish="length", completion_tokens=100)
    fake_glm.stream(content="second half.", finish="stop", completion_tokens=40)
    completion = run(fake_glm)
    assert completion.content == "first half, second half."
    assert completion.continuations == 1
    assert completion.usage.completion_tokens == 140
    assert fake_glm.requests[1]["messages"] == [
        *MESSAGES,
        {"role": "assistant", "content": "first half, "},
        {"role": "user", "content": CONTINUE_PROMPT},
    ]


def test_reply_cut_off_while_reasoning_extends_the_open_reasoning_turn(fake_glm):
    fake_glm.stream(reasoning="step one, ", finish="length")
    fake_glm.stream(reasoning="step two, ", finish="length")
    fake_glm.stream(reasoning="step three", content="answer, cut", finish="length")
    fake_glm.stream(content=" off")
    completion = run(fake_glm)
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


def test_length_with_a_tool_call_is_not_continued(fake_glm):
    fake_glm.stream(tool_calls=(("answer", '{"city": "Par'),), finish="length")
    completion = run(fake_glm)
    assert completion.finish_reason is FinishReason.LENGTH
    assert completion.continuations == 0
    assert len(fake_glm.requests) == 1


def test_continuation_that_no_longer_fits_returns_the_partial_output(fake_glm):
    fake_glm.stream(content="long answer", finish="length", prompt_tokens=140_020, completion_tokens=122_124)
    fake_glm.status(400, CONTEXT_ERROR)
    fake_glm.status(400, CONTEXT_ERROR)
    completion = run(fake_glm)
    assert completion.content == "long answer"
    assert completion.finish_reason is FinishReason.LENGTH
    assert completion.continuations == 0
    assert [(a.segment, a.outcome) for a in completion.attempts] == [
        (0, AttemptOutcome.COMPLETED),
        (1, AttemptOutcome.CONTEXT_OVERFLOW),
        (-1, AttemptOutcome.CONTEXT_OVERFLOW),
    ]


def test_prompt_that_fills_the_context_is_rejected(fake_glm):
    fake_glm.status(400, CONTEXT_ERROR)
    fake_glm.status(400, CONTEXT_ERROR)
    with pytest.raises(GlmContextExhausted):
        run(fake_glm)
    assert [r["max_tokens"] for r in fake_glm.requests] == [GLM_MAX_OUTPUT_TOKENS, 1]


def test_continuations_stop_at_policy_bound(fake_glm):
    for part in ("a", "b", "c"):
        fake_glm.stream(content=part, finish="length")
    completion = run(fake_glm, policy=LLMPolicy(max_continuations=2))
    assert completion.content == "abc"
    assert completion.finish_reason is FinishReason.LENGTH
    assert len(fake_glm.requests) == 3
