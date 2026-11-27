import asyncio

import pytest

from skyrl_train.inference_engines.request_deadline import finish_partial_on_timeout


@pytest.mark.asyncio
async def test_deadline_returns_the_live_response_without_cancelling_its_collector():
    pending = asyncio.get_running_loop().create_future()
    partial = {
        "token_ids": [7, 11],
        "logprobs": [-0.25, -0.75],
        "routed_experts": [[1, 2], [2, 3]],
        "policy_steps": [3, 3],
        "finish_reason": "abort",
    }

    async def finish():
        pending.set_result(partial)
        return True

    output, deadline_finished = await finish_partial_on_timeout(pending, 0, finish)
    assert output == partial
    assert deadline_finished
    assert not pending.cancelled()


@pytest.mark.asyncio
async def test_natural_completion_wins_the_deadline_race():
    pending = asyncio.get_running_loop().create_future()
    pending.set_result({"token_ids": [7], "finish_reason": "stop"})

    async def finish():
        raise AssertionError("A completed request must not be aborted")

    output, deadline_finished = await finish_partial_on_timeout(pending, 0, finish)
    assert output == {"token_ids": [7], "finish_reason": "stop"}
    assert not deadline_finished


@pytest.mark.asyncio
async def test_unstarted_request_remains_cancellable_without_inventing_a_response():
    pending = asyncio.get_running_loop().create_future()
    checked = asyncio.Event()

    async def finish():
        checked.set()
        return False

    caller = asyncio.create_task(finish_partial_on_timeout(pending, 0, finish))
    await checked.wait()
    assert not caller.done()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert pending.cancelled()


@pytest.mark.asyncio
async def test_failed_abort_does_not_leave_a_generation_running():
    pending = asyncio.get_running_loop().create_future()

    async def finish():
        raise ConnectionError("engine disconnected")

    with pytest.raises(ConnectionError, match="engine disconnected"):
        await finish_partial_on_timeout(pending, 0, finish)
    assert pending.cancelled()
