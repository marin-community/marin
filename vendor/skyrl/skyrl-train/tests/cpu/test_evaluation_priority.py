"""Priority evaluation shares live rollout workers and fences serving-policy mutations."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.rollouts.workers import RolloutWorkerPool, RolloutWorkerResources


@pytest.mark.asyncio
async def test_priority_session_finishes_while_existing_and_new_training_groups_are_in_flight():
    started = asyncio.Event()
    release = asyncio.Event()
    sessions = []

    async def start(**kwargs):
        sessions.append(kwargs["eval_step"])

    async def stop():
        sessions.append("stopped")

    actor = SimpleNamespace(
        start_eval_session=SimpleNamespace(remote=start), stop_eval_session=SimpleNamespace(remote=stop)
    )
    pool = RolloutWorkerPool(
        SimpleNamespace(config=OmegaConf.create({"trainer": {"rollout_buffer": {"eval_drain": "priority"}}})),
        RolloutWorkerResources(1, 1, 1, 30, 0),
    )
    pool._actors = [actor]

    async def training(_actor):
        started.set()
        await release.wait()
        return "training-complete"

    async def evaluate(_actor):
        return "evaluation-complete"

    first = asyncio.create_task(pool._dispatch("train", training))
    await started.wait()
    try:
        await pool.start_eval_session(run_name="priority", eval_step=3)
        started.clear()
        second = asyncio.create_task(pool._dispatch("train", training))
        await started.wait()
        assert await pool._dispatch("eval", evaluate) == "evaluation-complete"
        await pool.stop_eval_session()
        assert not first.done() and not second.done()
        assert pool._pending == [2]
        assert sessions == [3, "stopped"]
    finally:
        release.set()
        await first
        await second
