# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trial execution budgets exclude a saturated engine's admission queue."""

import asyncio

import pytest
from rigging.inference_clock import InferenceClock
from taskcompendium.grading_result import Outcome

from rolloutengine.models import TOTAL_TURN_TIMEOUT_STOP_REASON, GradedRolloutInterrupted
from rolloutengine.test_support import ReplayModel, arithmetic_task, engine, rollout_spec


@pytest.mark.parametrize("finish", [True, False])
async def test_saturated_engine_wait_does_not_spend_trial_or_turn_budget(monkeypatch, finish):
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = 0.0
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset)
    slot = asyncio.Semaphore(1)
    await slot.acquire()
    queued, admitted, answer = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clock = InferenceClock(1000)

    class QueuedModel(ReplayModel):
        async def complete(self, request):
            async with clock.queue():
                queued.set()
                await slot.acquire()
                clock.scheduled()
                admitted.set()
                try:
                    await answer.wait()
                    return await super().complete(request)
                finally:
                    slot.release()

    spec = rollout_spec(arithmetic_task())
    spec = spec.model_copy(
        update={
            "limits": spec.limits.model_copy(
                update={
                    "model_turn_timeout": 5,
                    "total_turn_timeout": 5,
                    "attempt_timeout": 10,
                }
            )
        }
    )

    async def run():
        with clock.scope():
            return await engine(QueuedModel([{"role": "assistant", "content": "12"}])).run_task(spec)

    pending = asyncio.create_task(run())
    await queued.wait()
    offset += 50  # All old execution deadlines would have expired; queue allowance has not.
    for _ in range(3):
        await asyncio.sleep(0)
    assert not pending.done()
    slot.release()
    admitted_wait = asyncio.create_task(admitted.wait())
    done, _ = await asyncio.wait({pending, admitted_wait}, return_when=asyncio.FIRST_COMPLETED)
    assert admitted_wait in done, "The trial expired before owning its engine slot"
    if finish:
        answer.set()
        result = await pending
        assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1)
        assert result.rollout.metrics["timing/inference_queue_wait"] >= 50
    else:
        offset += 6
        result = await pending
        assert result.rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON
        assert result.grade.status == Outcome.UNAVAILABLE
    assert clock.scheduled_requests == 1


async def test_never_scheduled_request_uses_separate_queue_allowance(monkeypatch):
    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = 0.0
    monkeypatch.setattr(loop, "time", lambda: real_time() + offset)
    queued = asyncio.Event()
    clock = InferenceClock(20)

    class UnscheduledModel(ReplayModel):
        async def complete(self, request):
            async with clock.queue():
                queued.set()
                await asyncio.Future()
            raise AssertionError("An unowned engine slot cannot produce a turn")

    async def run():
        with clock.scope():
            return await engine(UnscheduledModel([])).run_task(rollout_spec(arithmetic_task()))

    pending = asyncio.create_task(run())
    await queued.wait()
    offset += 21
    with pytest.raises(GradedRolloutInterrupted) as captured:
        await pending
    result = captured.value
    assert result.grade.status == Outcome.UNAVAILABLE
    assert result.rollout.failure is not None
    assert result.rollout.failure.exception_type == "InferenceAdmissionTimeout"
    assert result.rollout.metrics["inference/queue_timeouts"] == 1
    assert result.rollout.metrics["inference/scheduled_requests"] == 0
    assert not result.rollout.conversations[0].steps


async def test_transport_without_scheduling_notice_keeps_the_turn_deadline():
    clock = InferenceClock(20)

    class UntrackedModel(ReplayModel):
        async def complete(self, request):
            await asyncio.Future()

    spec = rollout_spec(arithmetic_task(), total_turn_timeout=0.05, model_turn_timeout=1, attempt_timeout=2)
    with clock.scope():
        result = await asyncio.wait_for(engine(UntrackedModel([])).run_task(spec), timeout=1)
    assert result.rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON
    assert result.grade.status == Outcome.UNAVAILABLE
    assert clock.scheduled_requests == 0
