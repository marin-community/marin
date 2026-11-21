# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from lib.taskforge.tests.resources.inference_fakes import HANG, Fails, ScriptedProvider, chunk, request, segment
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import GradedRollout, TaskRolloutSpec, TraceMode
from rolloutengine.test_support import arithmetic_task, rollout_spec

from taskforge import wiring
from taskforge.effects.issues import Restart
from taskforge.llm import glm_provider
from taskforge.llm.client import AttemptOutcome, ContextExhausted, ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.glm_provider import FixedEndpoint, GlmEndpoint, Pool
from taskforge.llm.policy import GLM_5_3, LLMPolicy, RetryPolicy
from taskforge.model.llm import FinishReason
from taskforge.model.review import Cause
from taskforge.policy import load_policy
from taskforge.resources.rollouts import RolloutResource, SolverRollout, run_rollout
from taskforge.stages.classify import trial_cause
from taskforge.wiring import provider_policy, solver_turn

DEFAULTS = Path(__file__).parents[2] / "src/taskforge/policy/defaults"
TOOL_STALL = Fails(AttemptOutcome.STALLED, (chunk(role="assistant"), chunk(tool_calls=[{"index": 0, "id": "c"}])))
"""A stall after a tool call opened: the server buffers its arguments."""


class Time:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.waits.append(delay)
        self.now += delay


def client(
    provider: ScriptedProvider,
    time: Time | None = None,
    *,
    wait: float = 3600,
    backoff: tuple[float, float, float] = (1, 120, 2),
    generation_timeout: float | None = None,
) -> ModelClient:
    retry = RetryPolicy(wait, *backoff, generation_timeout=generation_timeout)
    if time is None:
        return ModelClient(provider, GLM_5_3, retry, role="solver")
    return ModelClient(provider, GLM_5_3, retry, role="solver", clock=time.clock, sleep=time.sleep)


def unavailable(outcome: AttemptOutcome, *events: str) -> Fails:
    return Fails(outcome, events, 503 if outcome is AttemptOutcome.RETRYABLE_STATUS else None)


async def test_busy_provider_waits_past_eight_attempts_with_capped_backoff() -> None:
    outcomes = (
        AttemptOutcome.RETRYABLE_STATUS,
        AttemptOutcome.STALLED,
        AttemptOutcome.ABORTED,
        AttemptOutcome.INCOMPLETE_STREAM,
    )
    provider = ScriptedProvider((*(unavailable(outcome) for outcome in outcomes * 3), segment("answer")))
    time = Time()
    result = await client(provider, time).complete((), LLMPolicy())
    assert result.content == "answer" and len(provider.bodies) == 13
    assert [attempt.outcome for attempt in result.attempts] == [*outcomes * 3, AttemptOutcome.COMPLETED]
    assert max(time.waits) == 120 and time.waits[:4] == [1, 2, 4, 8]


async def test_busy_provider_stops_at_window_not_attempt_count() -> None:
    provider = ScriptedProvider((unavailable(AttemptOutcome.RETRYABLE_STATUS),) * 10)
    time = Time()
    with pytest.raises(ModelUnavailable, match="5s") as raised:
        await client(provider, time, wait=5, backoff=(1, 60, 2)).complete((), LLMPolicy())
    assert time.now == 5 and time.waits == [1, 2, 2]
    assert len(provider.bodies) == len(raised.value.attempts) == 3


@pytest.mark.parametrize(
    ("cut", "served"),
    [
        (unavailable(AttemptOutcome.INCOMPLETE_STREAM, chunk(content="partial")), True),
        (unavailable(AttemptOutcome.STREAM_ERROR, chunk(content="partial")), True),
        (unavailable(AttemptOutcome.STALLED, chunk(content="partial")), False),
    ],
    ids=["partial-stream", "stream-error", "stall-in-content"],
)
async def test_a_stream_cut_after_streaming_restarts_the_window_and_a_stall_does_not(cut: Fails, served: bool) -> None:
    """A replica reclaimed mid-stream shows the provider was available; an engine gone silent does not."""
    busy = unavailable(AttemptOutcome.RETRYABLE_STATUS)
    provider = ScriptedProvider((busy, busy, cut, busy, segment("answer")))
    model = client(provider, Time(), wait=5, backoff=(1, 60, 2))
    if served:
        assert (await model.complete((), LLMPolicy())).content == "answer"
        return
    with pytest.raises(ModelUnavailable):
        await model.complete((), LLMPolicy())
    assert len(provider.bodies) == 3


@pytest.mark.parametrize("pool_returns", [True, False], ids=["recovers", "window-closes"])
async def test_an_empty_pool_is_polled_instead_of_sent_to(pool_returns: bool) -> None:
    empty = [True] * 3 + ([False] if pool_returns else [True] * 100)
    provider = ScriptedProvider((unavailable(AttemptOutcome.RETRYABLE_STATUS), segment("answer")), empty_pool=empty)
    model = client(provider, Time(), wait=60, backoff=(1, 4, 2))
    if pool_returns:
        assert (await model.complete((), LLMPolicy())).content == "answer"
    else:
        with pytest.raises(ModelUnavailable):
            await model.complete((), LLMPolicy())
    assert len(provider.bodies) == (2 if pool_returns else 1)


async def test_consecutive_stalls_inside_a_tool_call_return_degenerate_output() -> None:
    provider = ScriptedProvider((TOOL_STALL, unavailable(AttemptOutcome.STALLED), TOOL_STALL, TOOL_STALL))
    result = await client(provider, Time()).complete((), LLMPolicy())
    assert (result.content, result.tool_calls, result.finish_reason) == ("", (), FinishReason.LENGTH)
    assert len(result.attempts) == len(provider.bodies) == 4  # a stall in no tool call resets the count


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (Fails(AttemptOutcome.REJECTED, status=401), ModelRequestRejected),
        (Fails(AttemptOutcome.CONTEXT_OVERFLOW, status=400), ContextExhausted),
    ],
)
async def test_a_rejection_is_neither_waited_nor_retried(failure: Fails, error: type[Exception]) -> None:
    provider = ScriptedProvider((failure, segment("never")))
    time = Time()
    with pytest.raises(error):
        await client(provider, time).complete((), LLMPolicy())
    assert len(provider.bodies) == 1 and not time.waits


SPEC = rollout_spec(arithmetic_task(), trace=TraceMode.MESSAGES, model_turn_timeout=5.0)


def solver(model: ModelClient) -> RolloutResource:
    """The real solver path: RolloutEngine's shell harness calling ``solver_turn`` on ``model``."""
    engine = RolloutEngine({}, {"shell": ShellHarness(command_timeout=None)})
    return RolloutResource(engine, partial(solver_turn, model, LLMPolicy()), "solver")


async def rollout(model: ModelClient, spec: TaskRolloutSpec = SPEC) -> GradedRollout:
    return await run_rollout(SolverRollout(spec, 0, 0), solver(model), notify=lambda _: None)


@pytest.mark.parametrize(
    ("script", "generation_timeout"),
    [
        ((Fails(AttemptOutcome.CONTEXT_OVERFLOW, status=400),), None),
        ((HANG,), 0.05),
        ((TOOL_STALL, TOOL_STALL), None),
    ],
    ids=["context-overflow", "turn-deadline", "degenerate-tool-call"],
)
async def test_a_solver_turn_that_cannot_finish_is_an_unsolved_trial_and_is_not_retried(
    script: tuple[Fails | str, ...], generation_timeout: float | None
) -> None:
    """The trial ends at the generation limit with no transcript: counted in the band, never solved or retried."""
    provider = ScriptedProvider(script)
    result = await rollout(client(provider, generation_timeout=generation_timeout))
    assert result.rollout.stop_reason == "length"
    assert trial_cause(result, SPEC.task) is Cause.GENERATION_LIMIT
    assert len(provider.bodies) == len(script)


async def test_a_solver_transport_stall_is_retried_by_the_model_client() -> None:
    provider = ScriptedProvider((unavailable(AttemptOutcome.STALLED, chunk(content="1")), segment("12")))
    result = await rollout(client(provider, Time()))
    assert result.grade.reward == 1 and trial_cause(result, SPEC.task) is None
    assert len(provider.bodies) == 2


@pytest.mark.parametrize(
    "outage",
    [
        Fails(AttemptOutcome.ROUTE_MISSING, status=404, detail="no route for model glm-5.3"),
        Fails(AttemptOutcome.RETRYABLE_STATUS, status=503, detail="no workers"),
        Fails(AttemptOutcome.TRANSPORT_ERROR, detail="no endpoint: relay not registered"),
    ],
    ids=["no-route", "5xx", "unregistered-relay"],
)
@pytest.mark.parametrize("recovers", [True, False])
async def test_a_solver_provider_outage_is_waited_out_then_restarts_never_a_trial(outage: Fails, recovers: bool) -> None:
    provider = ScriptedProvider((outage, outage, segment("12") if recovers else outage))
    model = client(provider, Time(), wait=3, backoff=(1, 1, 1), generation_timeout=60)
    if recovers:
        result = await rollout(model)
        assert result.grade.reward == 1 and trial_cause(result, SPEC.task) is None
        return
    with pytest.raises(Restart, match="interrupted during model"):
        await rollout(model)


async def test_an_outage_that_outlasts_the_turn_deadline_restarts_never_a_trial() -> None:
    """After a failure the generation bound is off: the engine's model-turn deadline ends the attempt as an outage."""
    provider = ScriptedProvider((unavailable(AttemptOutcome.RETRYABLE_STATUS), HANG))
    model = client(provider, backoff=(0.001, 0.001, 1), generation_timeout=0.05)
    with pytest.raises(Restart, match="interrupted during model"):
        await rollout(
            model, SPEC.model_copy(update={"limits": SPEC.limits.model_copy(update={"model_turn_timeout": 0.3})})
        )
    assert len(provider.bodies) == 2


async def test_cancelling_a_solver_rollout_propagates() -> None:
    provider = ScriptedProvider((HANG,))
    running = asyncio.create_task(rollout(client(provider)))
    await provider.streaming.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


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

    monkeypatch.setattr(glm_provider.httpx, "AsyncClient", MockClient)
    source = FixedEndpoint(GlmEndpoint("http://fake/v1", "fake", Pool.BULK))
    policy = load_policy(DEFAULTS)
    monkeypatch.setattr(wiring, "get_job_info", lambda: SimpleNamespace(controller_address="http://unused"))
    async with wiring.runtime(policy, source, "fake") as (_, resources):
        for role in ("author", "solver", "adversary"):
            await resources.inference_for(role).complete(request())
        assert resources.inference_for("author") is resources.inference
        assert resources.inference_for("author").fingerprint() != resources.inference_for("adversary").fingerprint()
    assert [body["model"] for body in bodies] == ["deepseek-v4.1-flash", "glm-5.3", "glm-5.3"]
    assert all("max_tokens" not in body and "max_completion_tokens" not in body for body in bodies)


def test_trial_timeouts_outlast_declared_provider_window() -> None:
    resolved = provider_policy(load_policy(DEFAULTS))
    assert resolved.setting("solver.turn_timeout", float) == 4200
    assert resolved.setting("solver.attempt_timeout", float) == 8400


@pytest.mark.parametrize(
    ("overlay", "message"),
    [
        ({"solver.turn_timeout": 1800.0}, "must cover"),
        ({"solver.attempt_timeout": 3600.0}, "must cover"),
        (
            {"provider.unavailable_wait": 10.0, "solver.turn_timeout": 650.0, "solver.attempt_timeout": 1300.0},
            "less its margin",
        ),
    ],
    ids=["turn-under-window", "attempt-under-two-turns", "generation-bound-under-a-stall"],
)
def test_inconsistent_provider_deadlines_are_rejected(overlay: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        provider_policy(load_policy(DEFAULTS).overlaid(overlay))
