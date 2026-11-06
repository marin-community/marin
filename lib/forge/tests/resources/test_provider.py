# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import httpx
import pytest
from taskforge.llm import glm_client
from taskforge.llm.client import Attempt, AttemptOutcome, ModelRequestRejected, ModelUnavailable
from taskforge.llm.glm_client import FixedEndpoint, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig

from forge.policy import load_policy
from forge.resources.provider import EndpointLimitedClient, WaitingBackend
from forge.wiring import provider_policy
from tests.resources.test_inference import Backend, provider_completion


class Time:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.waits.append(delay)
        self.now += delay


def unavailable(outcome: AttemptOutcome = AttemptOutcome.RETRYABLE_STATUS) -> ModelUnavailable:
    return ModelUnavailable("busy", (Attempt(0, outcome, 0, 0, 0, 503, "busy", ()),))


async def test_busy_provider_waits_past_eight_attempts_with_capped_backoff() -> None:
    failures = [
        unavailable(outcome)
        for outcome in (
            AttemptOutcome.RETRYABLE_STATUS,
            AttemptOutcome.STALLED,
            AttemptOutcome.ABORTED,
            AttemptOutcome.INCOMPLETE_STREAM,
        )
    ] * 3
    backend = Backend((*failures, provider_completion()))
    time = Time()
    waiting = WaitingBackend(backend, 3600, 1, 120, 2, clock=time.clock, sleep=time.sleep)
    result = await waiting.complete((), LLMPolicy())
    assert result.content == "answer" and len(backend.calls) == 13
    assert len(result.attempts) == 12
    assert max(time.waits) == 60 and time.waits[:4] == [1, 2, 4, 8]


async def test_busy_provider_stops_at_window_not_attempt_count() -> None:
    backend = Backend((unavailable(),) * 10)
    time = Time()
    waiting = WaitingBackend(backend, 5, 1, 60, 2, clock=time.clock, sleep=time.sleep)
    with pytest.raises(ModelUnavailable, match="5s"):
        await waiting.complete((), LLMPolicy())
    assert time.now == 5 and time.waits == [1, 2, 2]
    assert len(backend.calls) == 3


async def test_provider_400_is_not_waited_or_retried() -> None:
    rejection = ModelRequestRejected(400, "maximum context length", ())
    backend = Backend((rejection,))
    time = Time()
    with pytest.raises(ModelRequestRejected) as caught:
        await WaitingBackend(backend, 3600, 1, 60, 2, clock=time.clock, sleep=time.sleep).complete((), LLMPolicy())
    assert caught.value is rejection and not time.waits


async def test_endpoint_request_body_has_no_client_output_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 200000, "total_tokens": 200001}},
        ]
        body = "".join(
            "data: "
            + json.dumps({"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test", **chunk})
            + "\n\n"
            for chunk in chunks
        )
        return httpx.Response(200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    original = httpx.AsyncClient

    class MockClient(original):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(glm_client.httpx, "AsyncClient", MockClient)
    provider = LLMProviderConfig("test", 1, 3600, 1, 60, 2, 600, 10, 1)
    source = FixedEndpoint(GlmEndpoint("http://fake/v1", "fake", Pool.BULK))
    async with EndpointLimitedClient(source, provider) as client:
        result = await client.complete(({"role": "user", "content": "hello"},), LLMPolicy(max_tokens=1))
    assert result.content == "ok"
    assert len(bodies) == 1
    assert "max_tokens" not in bodies[0] and "max_completion_tokens" not in bodies[0]


def test_trial_timeouts_outlast_declared_provider_window(capsys: pytest.CaptureFixture[str]) -> None:
    policy = load_policy(Path(__file__).parents[2] / "src/forge/policy/defaults")
    resolved = provider_policy(policy)
    assert resolved.setting("solver.turn_timeout", float) == 4200
    assert resolved.setting("solver.attempt_timeout", float) == 8400
    output = capsys.readouterr().out
    assert "provider.unavailable_wait declared=3600.0 resolved=3600.0" in output
    assert "solver.turn_timeout declared=1800.0 resolved=4200.0" in output
