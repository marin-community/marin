# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from lib.taskforge.tests.resources.inference_fakes import (
    HANG,
    Act,
    Fails,
    LoopTime,
    ScriptedProvider,
    Time,
    chunk,
    report,
    request,
    segment,
)
from pydantic import TypeAdapter
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import GradedRollout, ModelRequest, Rollout, TaskRolloutSpec, TraceMode
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge import wiring
from taskforge.effects.events import Resolved
from taskforge.effects.issues import Restart
from taskforge.llm import glm_provider
from taskforge.llm.client import (
    CONFIRMATION_INTERVAL,
    AttemptFailed,
    AttemptOutcome,
    ContextExhausted,
    ModelClient,
    ModelRequestRejected,
    ModelUnavailable,
    Progress,
    Segment,
)
from taskforge.llm.glm_provider import FixedEndpoint, GlmEndpoint, Pool
from taskforge.llm.policy import GLM_5_3, LLMPolicy, ReasoningEffort, RetryPolicy
from taskforge.lowering import lowering_settings, requirements
from taskforge.model.content_hash import json_value
from taskforge.model.llm import FinishReason, ToolCall
from taskforge.model.review import Cause, TrialOutcome, TrialResult
from taskforge.policy import load_policy
from taskforge.resources.rollouts import RolloutResource, SolverRollout, run_rollout
from taskforge.resources.sandbox.factories import shellsim_factory
from taskforge.stages.classify import replayable_grade, trial_cause
from taskforge.stages.rules import solved
from taskforge.wiring import provider_policy, retry_policy, solver_turn

DEFAULTS = Path(__file__).parents[2] / "src/taskforge/policy/defaults"
TOOL_STALL = Fails(AttemptOutcome.STALLED, (chunk(role="assistant"), chunk(tool_calls=[{"index": 0, "id": "c"}])))
"""A stall after a tool call opened: the server buffers its arguments."""


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


# Regression: a second silent stall inside a tool call became an empty `length` completion, a failed trial.
async def test_stalls_inside_a_tool_call_are_waited_out_as_an_outage() -> None:
    """Silence cannot tell a model buffering degenerate arguments from a fleet that went away, so it is never a trial."""
    provider = ScriptedProvider((TOOL_STALL, TOOL_STALL, TOOL_STALL, segment("answer")))
    result = await client(provider, Time()).complete((), LLMPolicy())
    assert (result.content, len(provider.bodies)) == ("answer", 4)


# Regression: a role header and a close, eight times, kept a call alive past a 3 s availability window.
# A tool-call frame with only an index or empty arguments also counted as generated text.
@pytest.mark.parametrize(
    "envelope",
    [
        chunk(role="assistant"),
        chunk(tool_calls=[{"index": 0}]),
        chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "", "arguments": ""}}]),
    ],
    ids=["role", "tool-index", "empty-tool-arguments"],
)
async def test_a_stream_cut_before_generating_does_not_renew_the_availability_window(envelope: str) -> None:
    header = unavailable(AttemptOutcome.INCOMPLETE_STREAM, envelope)
    provider = ScriptedProvider((header,) * 8 + (segment("answer"),))
    time = Time()
    with pytest.raises(ModelUnavailable):
        await client(provider, time, wait=3, backoff=(1, 1, 1)).complete((), LLMPolicy())
    assert (time.now, len(provider.bodies)) == (3, 3)


# Regression: one generated character then EOF, repeated, renewed the window until 65 POSTs.
async def test_a_call_renews_its_availability_window_once() -> None:
    cut = unavailable(AttemptOutcome.INCOMPLETE_STREAM, chunk(content="1"))
    provider = ScriptedProvider((cut,) * 64 + (segment("answer"),))
    time = Time()
    with pytest.raises(ModelUnavailable):
        await client(provider, time, wait=3, backoff=(1, 1, 1)).complete((), LLMPolicy())
    assert (time.now, len(provider.bodies)) == (3, 3)


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
async def test_the_profile_output_limit_caps_a_token_limit_the_request_sets(field: str) -> None:
    provider = ScriptedProvider((segment("12"),))
    capped = replace(GLM_5_3, name="glm-5.3/output-4", max_output_tokens=4)
    model = ModelClient(provider, capped, RetryPolicy(60, 1, 1, 1), role="solver")
    await solver_turn(model, LLMPolicy(), ModelRequest(({"role": "user", "content": "6+6"},), {field: 64}))
    assert provider.bodies[0][field] == 4


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


async def test_a_solver_context_overflow_is_an_unsolved_trial_and_is_not_retried() -> None:
    """The trial ends at the generation limit with no transcript: counted in the band, never solved or retried."""
    provider = ScriptedProvider((Fails(AttemptOutcome.CONTEXT_OVERFLOW, status=400),))
    result = await rollout(client(provider))
    assert result.rollout.stop_reason == "length"
    assert trial_cause(result, SPEC.task) is Cause.GENERATION_LIMIT
    assert len(provider.bodies) == 1


GENERATION_BOUND = 100.0
TURN = SPEC.model_copy(update={"limits": SPEC.limits.model_copy(update={"model_turn_timeout": GENERATION_BOUND + 60})})


def generates(time: LoopTime, seconds: float) -> Act:
    """A stream that generates a token every second for ``seconds``, then finishes."""

    async def act(progress: Progress) -> Segment:
        for _ in range(int(seconds)):
            await time.sleep(1)
            report(progress, chunk(content="x"))
        return segment("12")

    return act


def bounded(provider: ScriptedProvider, time: LoopTime, wait: float = 3600) -> ModelClient:
    retry = RetryPolicy(wait, 1, 1, 1, generation_timeout=GENERATION_BOUND)
    return ModelClient(provider, GLM_5_3, retry, role="solver", clock=time.clock, sleep=time.sleep)


# Regression: after any earlier fault the bound was off, so a long healthy generation ate stage retries.
@pytest.mark.parametrize("before", [(), (unavailable(AttemptOutcome.RETRYABLE_STATUS),)], ids=["first", "after-a-fault"])
async def test_a_generation_sustained_past_its_bound_is_a_time_limited_trial(before: tuple[Fails, ...]) -> None:
    """Recorded as reaching the solver time limit, not the generation limit, and counted unsolved."""
    time = LoopTime()
    provider = ScriptedProvider((*before, generates(time, 10 * GENERATION_BOUND)))
    result = await rollout(bounded(provider, time), TURN)
    trial = TrialResult(0, result.grade, trial_cause(result, SPEC.task))
    assert trial.outcome is TrialOutcome.TIME_LIMIT and not solved(trial)
    assert len(provider.bodies) == len(before) + 1
    # The first token is at second 1; the first token past the bound is confirmed one interval later.
    assert time.offset == len(before) + 1 + GENERATION_BOUND + CONFIRMATION_INTERVAL


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


# Regression: an unknown 400 wording was restarted twice by the rollout wrapper and the coordinator.
async def test_a_solver_request_the_server_rejects_holds_its_slot_at_once_with_its_status_and_reason() -> None:
    reason = "the prompt exceeds the input token budget"
    provider = ScriptedProvider((Fails(AttemptOutcome.REJECTED, status=400, detail=reason), segment("12")))
    result = await rollout(client(provider, Time()))
    assert trial_cause(result, SPEC.task) is Cause.HELD and "RequestRejected: " in (result.grade.error or "")
    assert f"HTTP 400: {reason}" in (result.grade.error or "") and len(provider.bodies) == 1


# Regression: a refusal on turn two recorded an empty rollout, dropping the paid first step.
async def test_a_later_turn_refusal_keeps_the_paid_rollout_prefix_and_its_usage() -> None:
    shell = ToolCall("c1", "shell", json.dumps({"command": "echo hi"}))
    provider = ScriptedProvider(
        (segment("", (shell,), FinishReason.TOOL_CALLS), Fails(AttemptOutcome.REJECTED, status=400))
    )
    machine = rollout_spec(
        arithmetic_task(environment_requirements=requirements(image=None)),
        machine=machine_runtime(backend="shellsim"),
        trace=TraceMode.MESSAGES,
        model_turn_timeout=5.0,
    )
    engine = RolloutEngine({"shellsim": shellsim_factory()}, {"shell": ShellHarness(command_timeout=None)})
    resource = RolloutResource(engine, partial(solver_turn, client(provider), LLMPolicy()), "solver")
    result = await run_rollout(SolverRollout(machine, 0, 0), resource, notify=lambda _: None)
    assert trial_cause(result, machine.task) is Cause.HELD and len(provider.bodies) == 2
    (step,) = result.rollout.conversations[0].steps
    assert step.turn.metadata["usage"]["prompt_tokens"] == 10 and step.turn.metadata["usage"]["completion_tokens"] == 5


# Regression: a model whose stop text was "held" and whose answer graded 1.0 was a missing slot.
def test_no_model_output_reads_as_a_recorded_refusal() -> None:
    answered = GradedRollout(Rollout("arithmetic:0:0", (), "held"), GradeResult(Outcome.GRADED, 1.0))
    assert trial_cause(answered, SPEC.task) is None
    assert replayable_grade(Resolved(SolverRollout(SPEC, 0, 0), (), answered))


async def test_cancelling_a_solver_rollout_propagates() -> None:
    provider = ScriptedProvider((HANG,))
    running = asyncio.create_task(rollout(client(provider)))
    await provider.streaming.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


class Trickle(httpx.AsyncByteStream):
    """An error body of seven bytes, each three virtual seconds after the last: 21 seconds in all."""

    def __init__(self, time: LoopTime) -> None:
        self.time = time

    async def __aiter__(self):  # type: ignore[override]
        for _ in range(7):
            await self.time.sleep(3)
            yield b" "


# Regression: the end-to-end trickle test showed a body was cut, but not at which deadline: a hook bound of
# 750 ms in place of the configured 50 ms still passed. The configured stall timeout bounds the whole body.
@pytest.mark.parametrize(("stall_timeout", "drained"), [(10.0, False), (150.0, True)])
async def test_an_error_body_is_read_whole_within_the_configured_stall_timeout(
    stall_timeout: float, drained: bool
) -> None:
    time = LoopTime()
    source = FixedEndpoint(GlmEndpoint("http://fake/v1", "fake", Pool.HIGH))
    provider = glm_provider.GlmProvider(source, "glm-5.3", stall_timeout, 1)
    request = httpx.Request("POST", "http://fake/v1/chat/completions")
    response = httpx.Response(503, stream=Trickle(time), request=request)
    if drained:
        await provider._read_error_body(response)
        assert response.content == b" " * 7
    else:
        with pytest.raises(httpx.ReadTimeout, match=f"error body unread in {stall_timeout}s"):
            await provider._read_error_body(response)


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


def stock(time_limit: float, **settings: float) -> tuple[RetryPolicy, TaskRolloutSpec]:
    """The solver client's retry policy and the engine's limits when only ``solver.time_limit`` and ``settings``
    change."""
    policy = provider_policy(load_policy(DEFAULTS).overlaid({"solver.time_limit": time_limit, **settings}))
    return retry_policy(policy, "solver"), SPEC.model_copy(update={"limits": lowering_settings(policy).limits})


# Regression: the derived 12,681 s model-turn wall cut healthy requests the client still allowed.
@pytest.mark.parametrize("time_limit", [1800.0, 4199.0, 4200.0, 4201.0, 7200.0])
def test_the_time_limit_bounds_generation_under_the_stock_attempt_cap_and_no_model_turn_wall(time_limit: float) -> None:
    retry, spec = stock(time_limit)
    assert retry.generation_timeout == time_limit
    assert (spec.limits.model_turn_timeout, spec.limits.attempt_timeout) == (None, 8400.0)


def test_a_time_limit_that_bounds_nothing_or_outlasts_the_attempt_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"solver\.time_limit must be positive"):
        stock(0.0)
    with pytest.raises(ValueError, match=r"solver\.attempt_timeout must be finite and greater than solver\.time_limit"):
        stock(8401.0)


# Regression: at an equal cap and time limit the attempt cap fired first, so a time-limited trial was recorded as an
# unscored cap and the task held instead of counting the trial unsolved.
def test_a_time_limit_equal_to_the_attempt_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"solver\.attempt_timeout must be finite and greater than solver\.time_limit"):
        stock(8400.0)


def test_the_renamed_time_limit_key_names_what_it_became() -> None:
    with pytest.raises(ValueError, match=r"solver\.turn_timeout: renamed solver\.time_limit, which bounds"):
        load_policy(DEFAULTS).overlaid({"solver.turn_timeout": 4200.0})


def walled(provider: ScriptedProvider, time: LoopTime, retry: RetryPolicy) -> ModelClient:
    return ModelClient(provider, GLM_5_3, retry, role="solver", clock=time.clock, sleep=time.sleep)


def silent(time: LoopTime, seconds: float, then: Act | None = None) -> Act:
    """An attempt that generates nothing for ``seconds``, then runs ``then`` or reports a stall."""

    async def act(progress: Progress) -> Segment:
        await time.sleep(seconds)
        if then is None:
            raise AttemptFailed(AttemptOutcome.STALLED, f"no progress for {seconds}s")
        return await then(progress)

    return act


def busy_for(seconds: float) -> Act:
    """A 503 whose Retry-After asks for ``seconds``."""

    async def act(_: Progress) -> Segment:
        raise AttemptFailed(AttemptOutcome.RETRYABLE_STATUS, "busy", 503, retry_after=seconds)

    return act


def cut_after(time: LoopTime, seconds: float) -> Act:
    """A stream that generates a token every second for ``seconds``, then is cut by the transport."""

    async def act(progress: Progress) -> Segment:
        await generates(time, seconds - 1)(progress)
        await time.sleep(1)
        raise AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "connection reset")

    return act


# Regression: two availability windows around a cut stream, then a long last stream. The stock watchdog stops the
# attempt before any limit is proven, which restarts the stage; an explicit larger cap lets the client decide.
@pytest.mark.parametrize(
    ("last", "cap", "cause", "outcome"),
    [
        (3000, 8400.0, None, None),
        (5000, 8400.0, None, None),
        (3000, 20000.0, None, TrialOutcome.GRADED),
        (5000, 20000.0, Cause.TIME_LIMIT, TrialOutcome.TIME_LIMIT),
    ],
    ids=["answers-capped", "past-limit-capped", "answers-larger-cap", "past-limit-larger-cap"],
)
async def test_a_slot_past_its_attempt_cap_restarts_and_inside_it_the_client_decides(
    last: float, cap: float, cause: Cause | None, outcome: TrialOutcome | None
) -> None:
    time = LoopTime()
    retry, spec = stock(4200.0, **{"solver.attempt_timeout": cap})
    provider = ScriptedProvider((busy_for(3400), cut_after(time, 4100), busy_for(3400), generates(time, last)))
    if outcome is None:  # the second wait was still running at 8,400 s
        with pytest.raises(Restart, match="attempt"):
            await rollout(walled(provider, time, retry), spec)
        return
    result = await rollout(walled(provider, time, retry), spec)
    trial = TrialResult(0, result.grade, trial_cause(result, spec.task))
    assert (trial.cause, trial.outcome) == (cause, outcome)
    assert len(provider.bodies) == 4 and result.rollout.failure is None and (cause or result.grade.reward == 1)


# Regression: at a 12,000 s time limit a cut stream and an answer took two full generations.
async def test_a_long_time_limit_with_a_larger_explicit_cap_keeps_a_recovery_that_generates_twice() -> None:
    time = LoopTime()
    retry, spec = stock(12000.0, **{"solver.attempt_timeout": 30000.0})
    provider = ScriptedProvider((cut_after(time, 11980), generates(time, 11980)))
    result = await rollout(walled(provider, time, retry), spec)
    assert trial_cause(result, spec.task) is None and result.grade.reward == 1 and len(provider.bodies) == 2


@pytest.mark.parametrize("time_limit", [1800.0, 7200.0])
async def test_generation_sustained_past_the_time_limit_is_a_time_limited_trial_under_the_stock_cap(
    time_limit: float,
) -> None:
    time = LoopTime()
    retry, spec = stock(time_limit)
    provider = ScriptedProvider((generates(time, 2 * time_limit),))
    result = await rollout(walled(provider, time, retry), spec)
    assert trial_cause(result, spec.task) is Cause.TIME_LIMIT and len(provider.bodies) == 1
    assert time.offset == 1 + time_limit + CONFIRMATION_INTERVAL


# Regression: the engine's wall started before admission and the first token, so a delayed first token let it cut
# a healthy stream that the client would have finished inside its generation bound.
@pytest.mark.parametrize("time_limit", [1800.0, 7200.0])
async def test_a_delayed_first_token_with_continuing_progress_finishes_inside_the_time_limit(time_limit: float) -> None:
    time = LoopTime()
    retry, spec = stock(time_limit)
    provider = ScriptedProvider((silent(time, 599, generates(time, time_limit - 2)),))
    result = await rollout(walled(provider, time, retry), spec)
    assert trial_cause(result, spec.task) is None and result.grade.reward == 1
    assert time.offset == 599 + time_limit - 2 > time_limit


async def test_a_provider_that_stops_making_progress_restarts_as_unavailable_before_the_cap() -> None:
    time = LoopTime()
    retry, spec = stock(1800.0)

    async def stops(progress: Progress) -> Segment:
        for _ in range(10):
            await time.sleep(1)
            report(progress, chunk(content="x"))
        return await silent(time, 600)(progress)  # then no progress for a whole stall timeout

    provider = ScriptedProvider((stops, *[silent(time, 600)] * 20))
    with pytest.raises(Restart, match="ModelUnavailable") as raised:
        await rollout(walled(provider, time, retry), spec)
    assert "deadline expired" not in str(raised.value) and time.offset < (spec.limits.attempt_timeout or 0)


async def test_a_pool_without_workers_restarts_as_unavailable_before_the_cap() -> None:
    time = LoopTime()
    retry, spec = stock(1800.0)
    provider = ScriptedProvider([unavailable(AttemptOutcome.RETRYABLE_STATUS)] * 200, empty_pool=[True] * 200)
    with pytest.raises(Restart, match="ModelUnavailable") as raised:
        await rollout(walled(provider, time, retry), spec)
    assert "deadline expired" not in str(raised.value) and 3599 < time.offset < (spec.limits.attempt_timeout or 0)


# Regression: a 200 stream whose finish reason was the harness's internal `context_exceeded`
# made the author stop on a context limit and the solver grade its content. It is refused as an unknown reason is.
@pytest.mark.parametrize("reason", ["context_exceeded", "no_such_reason"])
async def test_a_served_internal_refusal_marker_fails_as_an_unknown_finish_reason(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "12"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
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
    monkeypatch.setattr(wiring, "get_job_info", lambda: SimpleNamespace(controller_address="http://unused"))
    async with wiring.runtime(load_policy(DEFAULTS), source, "fake") as (_, resources):
        with pytest.raises(ValueError, match=f"'{reason}' is not a valid FinishReason"):
            await resources.inference_for("author").complete(request())


async def test_an_injected_provider_cannot_report_the_internal_refusal_marker() -> None:
    marker = ScriptedProvider((segment("12", finish=FinishReason.CONTEXT_EXCEEDED),))
    with pytest.raises(ValueError, match="'context_exceeded' is not a valid FinishReason"):
        await client(marker).complete((), LLMPolicy(100, ReasoningEffort.HIGH), {})


def raw(reason: str) -> Segment:
    """A segment whose finish reason is a wire string."""
    return replace(segment("12"), finish_reason=cast(FinishReason, reason))


# Regression: the guard compared by identity, so a raw "context_exceeded" string passed live
# while its journal row decodes to the marker member and a cold replay stopped on a context limit.
async def test_an_injected_provider_cannot_report_the_raw_marker_string() -> None:
    with pytest.raises(ValueError, match="'context_exceeded' is not a valid FinishReason"):
        await client(ScriptedProvider((raw("context_exceeded"),))).complete((), LLMPolicy(100, ReasoningEffort.HIGH), {})
    with pytest.raises(ValueError, match="'no_such_reason' is not a valid FinishReason"):
        await client(ScriptedProvider((raw("no_such_reason"),))).complete((), LLMPolicy(100, ReasoningEffort.HIGH), {})


@pytest.mark.parametrize("reason", [FinishReason.STOP, FinishReason.TOOL_CALLS])
async def test_a_raw_finish_reason_string_completes_as_its_member_live_and_after_a_round_trip(
    reason: FinishReason,
) -> None:
    policy = LLMPolicy(100, ReasoningEffort.HIGH)
    live = await client(ScriptedProvider((raw(reason.value),))).complete((), policy, {})
    typed = await client(ScriptedProvider((segment("12", finish=reason),))).complete((), policy, {})
    adapter = TypeAdapter(type(live))
    replayed = adapter.validate_python(json.loads(json.dumps(json_value(live))))  # as a journal row decodes it
    assert live.finish_reason is reason and replayed.finish_reason is reason
    assert replace(live, wall_time=0.0, attempts=()) == replace(typed, wall_time=0.0, attempts=())
