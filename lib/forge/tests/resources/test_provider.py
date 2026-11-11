# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from taskforge.llm import glm_client
from taskforge.llm.client import Attempt, AttemptOutcome, ModelRequestRejected, ModelUnavailable
from taskforge.llm.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig

from forge import wiring
from forge.policy import load_policy
from forge.resources.provider import WaitingBackend
from forge.wiring import provider_policy
from tests.resources.test_inference import Backend, provider_completion, request


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
    waiting = WaitingBackend(backend, 3600, 1, 120, 2, role="solver", clock=time.clock, sleep=time.sleep)
    result = await waiting.complete((), LLMPolicy())
    assert result.content == "answer" and len(backend.calls) == 13
    assert len(result.attempts) == 12
    assert max(time.waits) == 120 and time.waits[:4] == [1, 2, 4, 8]


async def test_busy_provider_stops_at_window_not_attempt_count(caplog: pytest.LogCaptureFixture) -> None:
    backend = Backend((unavailable(),) * 10)
    time = Time()
    waiting = WaitingBackend(backend, 5, 1, 60, 2, role="solver", clock=time.clock, sleep=time.sleep)
    with pytest.raises(ModelUnavailable, match=r"MODEL role=solver model=.* unavailable for 5s"):
        await waiting.complete((), LLMPolicy())
    assert time.now == 5 and time.waits == [1, 2, 2]
    assert f"MODEL role=solver model={backend.model} unavailable; 4.0s left" in caplog.text
    assert len(backend.calls) == 3


class StreamThenFail:
    """Each call streams for ``duration`` seconds, then fails as a reclaimed replica would."""

    model = "glm-5.3"

    def __init__(self, time: Time, duration: float, failures: int) -> None:
        self.time, self.duration, self.failures, self.calls = time, duration, failures, 0

    async def complete(self, *_: object) -> object:
        self.calls += 1
        self.time.now += self.duration
        if self.calls <= self.failures:
            raise ModelUnavailable(
                "reclaimed", (Attempt(0, AttemptOutcome.STALLED, 0, 0, self.duration, None, "cut", ('{"x":1}',)),)
            )
        return provider_completion()


async def test_failure_after_streaming_restarts_the_window() -> None:
    time = Time()
    backend = StreamThenFail(time, 2000, failures=3)
    waiting = WaitingBackend(backend, 3600, 1, 60, 2, role="solver", clock=time.clock, sleep=time.sleep)  # type: ignore[arg-type]
    result = await waiting.complete((), LLMPolicy())
    assert result.content == "answer" and backend.calls == 4
    assert time.waits == [1, 1, 1]


async def test_unregistered_relay_endpoint_is_waited_not_a_bug() -> None:
    backend = Backend((wiring.EndpointUnregistered("GLM relay /r has no registered endpoint for m"), provider_completion()))
    time = Time()
    waiting = WaitingBackend(backend, 3600, 1, 60, 2, role="author", clock=time.clock, sleep=time.sleep)
    result = await waiting.complete((), LLMPolicy())
    assert result.content == "answer" and len(backend.calls) == 2 and time.waits == [1]
    assert isinstance(wiring.EndpointUnregistered("x"), RuntimeError)


async def test_provider_400_is_not_waited_or_retried() -> None:
    rejection = ModelRequestRejected(400, "maximum context length", ())
    backend = Backend((rejection,))
    time = Time()
    with pytest.raises(ModelRequestRejected) as caught:
        await WaitingBackend(backend, 3600, 1, 60, 2, role="solver", clock=time.clock, sleep=time.sleep).complete(
            (), LLMPolicy()
        )
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
    async with GlmClient(source, provider, limit_output=False) as client:
        result = await client.complete(({"role": "user", "content": "hello"},), LLMPolicy(max_tokens=1))
    assert result.content == "ok"
    assert len(bodies) == 1
    assert "max_tokens" not in bodies[0] and "max_completion_tokens" not in bodies[0]
    policy = load_policy(Path(__file__).parents[2] / "src/forge/policy/defaults")
    monkeypatch.setattr(wiring, "get_job_info", lambda: SimpleNamespace(controller_address="http://unused"))
    async with wiring.runtime(policy, "S", source, "fake") as (_, resources):
        for role in ("author", "solver", "adversary"):
            await resources.inference_for(role).complete(request())
        assert resources.inference_for("author") is resources.inference
        assert resources.inference_for("author").fingerprint() != resources.inference_for("adversary").fingerprint()
    assert [body["model"] for body in bodies[1:]] == ["deepseek-v4.1-flash", "glm-5.3", "glm-5.3"]
    assert all("max_tokens" not in body for body in bodies)


def test_trial_timeouts_outlast_declared_provider_window() -> None:
    policy = load_policy(Path(__file__).parents[2] / "src/forge/policy/defaults")
    resolved = provider_policy(policy)
    assert resolved.setting("solver.turn_timeout", float) == 4200
    assert resolved.setting("solver.attempt_timeout", float) == 8400


@pytest.mark.parametrize("overlay", [{"solver.turn_timeout": 1800.0}, {"solver.attempt_timeout": 3600.0}])
def test_inconsistent_provider_deadlines_are_rejected(overlay: dict[str, float]) -> None:
    policy = load_policy(Path(__file__).parents[2] / "src/forge/policy/defaults").overlaid(overlay)
    with pytest.raises(ValueError, match="must cover"):
        provider_policy(policy)
