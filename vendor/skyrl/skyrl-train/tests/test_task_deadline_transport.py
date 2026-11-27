"""Deadline-driven partial return across the actor boundary, without a recipe cutoff."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from rigging.inference_clock import InferenceClock, execution_timeout
from rigging.inference_clock import PARTIAL_DEADLINE_STOP_REASON

from skyrl_train.inference_engines import ray_wrapped_inference_engine as transport
from skyrl_train.inference_engines.base import REQUEST_ID_HEADER
from skyrl_train.inference_engines.request_timing import RequestCompleted, RequestScheduled

QUEUE_ALLOWANCE = 200.0


@dataclass
class PendingActor:
    engine: transport.RayWrappedInferenceEngine
    generating: asyncio.Event
    aborted: asyncio.Event
    cancelled: list
    stream: AsyncGenerator[asyncio.Future[RequestScheduled | RequestCompleted[dict]], None]

    async def wait_for_generation(self, work: asyncio.Task) -> None:
        notification = asyncio.create_task(self.generating.wait())
        try:
            await asyncio.wait({work, notification}, return_when=asyncio.FIRST_COMPLETED)
            if work.done():
                await work
                raise AssertionError("The request finished before generation started")
        finally:
            notification.cancel()
            await asyncio.gather(notification, return_exceptions=True)


def pending_actor(monkeypatch, partial: dict | None) -> PendingActor:
    """Model only Ray's external RPC/stream boundary; keep the client and deadlines real."""
    loop = asyncio.get_running_loop()
    response = loop.create_future()
    generating, aborted = asyncio.Event(), asyncio.Event()
    actor = SimpleNamespace(
        generate_with_scheduling=object(), chat_completion_with_scheduling=object(), abort_active_partial_request=object()
    )
    request_id = None

    async def stream():
        notice = loop.create_future()
        notice.set_result(RequestScheduled())
        yield notice
        generating.set()
        yield response

    result_stream = stream()

    async def submit(method, payload):
        nonlocal request_id
        if method is actor.abort_active_partial_request:
            assert payload == request_id
            aborted.set()
            if partial is not None:
                response.set_result(RequestCompleted(partial))
            result = loop.create_future()
            result.set_result(partial is not None)
            return result
        request_id = payload["request_ids"][0] if "request_ids" in payload else payload["headers"][REQUEST_ID_HEADER]
        return result_stream

    cancelled = []
    monkeypatch.setattr(transport, "_submit_actor_task", submit)
    monkeypatch.setattr(transport.ray, "cancel", cancelled.append)
    engine = transport.RayWrappedInferenceEngine.__new__(transport.RayWrappedInferenceEngine)
    engine.inference_engine_actor = actor
    return PendingActor(engine, generating, aborted, cancelled, result_stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["generate", "chat_completion"])
async def test_partial_return_uses_the_remaining_trial_budget_after_an_earlier_turn(monkeypatch, endpoint):
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = 0.0
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset)
    sampled = (
        {"response_ids": [[7, 11]], "response_logprobs": [[-0.25, -0.75]],
         "routed_experts": [[[[1, 2]], [[2, 3]]]], "response_policy_steps": [[4, 4]]}
        if endpoint == "generate"
        else {"token_ids": [7, 11], "logprobs": {"content": [{"logprob": -0.25}, {"logprob": -0.75}]},
              "routed_experts": [[[1, 2]], [[2, 3]]], "policy_steps": [4, 4]}
    )
    partial = {"stop_reasons": ["abort"], **sampled} if endpoint == "generate" else {
        "choices": [{"finish_reason": "abort", **sampled}]
    }
    actor = pending_actor(monkeypatch, partial)
    clock = InferenceClock(QUEUE_ALLOWANCE)

    async def run():
        nonlocal offset
        with clock.scope():
            async with execution_timeout(100) as deadline:
                async with clock.queue():
                    clock.scheduled()
                offset += 80  # A previous model/tool phase already spent most of the trial budget.
                if endpoint == "generate":
                    result = await actor.engine.generate({"prompt_token_ids": [[1]], "session_ids": ["sample_0"]})
                else:
                    result = await actor.engine.chat_completion({"headers": {REQUEST_ID_HEADER: "sample"}})
                assert not deadline.expired()
                return result

    work = asyncio.create_task(run())
    try:
        await actor.wait_for_generation(work)
        offset += 11  # Twenty seconds remained, with half reserved for transport, rather than restarting all 100 seconds.
        await asyncio.wait_for(actor.aborted.wait(), 1)
        result = await work
        payload = result if endpoint == "generate" else result["choices"][0]
        assert {key: payload[key] for key in sampled} == sampled
        if endpoint == "generate":
            assert result["stop_reasons"] == [PARTIAL_DEADLINE_STOP_REASON]
        else:
            assert payload["finish_reason"] == PARTIAL_DEADLINE_STOP_REASON
    finally:
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        await actor.stream.aclose()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_turn_into_a_partial_response(monkeypatch):
    actor = pending_actor(monkeypatch, {"stop_reasons": ["abort"], "response_ids": [[7]]})
    clock = InferenceClock(QUEUE_ALLOWANCE)

    async def run():
        with clock.scope():
            async with execution_timeout(100):
                return await actor.engine.generate({"prompt_token_ids": [[1]], "session_ids": ["sample_0"]})

    work = asyncio.create_task(run())
    await actor.wait_for_generation(work)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert not actor.aborted.is_set()
    assert actor.cancelled == [actor.stream]
    await actor.stream.aclose()


@pytest.mark.asyncio
async def test_empty_active_request_remains_unavailable_at_the_execution_deadline(monkeypatch):
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = 0.0
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset)
    actor = pending_actor(monkeypatch, None)
    clock = InferenceClock(QUEUE_ALLOWANCE)

    async def run():
        with clock.scope():
            async with execution_timeout(100):
                return await actor.engine.generate({"prompt_token_ids": [[1]], "session_ids": ["sample_0"]})

    work = asyncio.create_task(run())
    try:
        await actor.wait_for_generation(work)
        offset += 71
        await asyncio.wait_for(actor.aborted.wait(), 1)
        assert not work.done()
        offset += 30
        with pytest.raises(TimeoutError):
            await work
    finally:
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        await actor.stream.aclose()
