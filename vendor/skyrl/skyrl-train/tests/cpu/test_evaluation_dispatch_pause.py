"""Evaluation admission pauses preserve pending prompts in restart state."""

import asyncio
from types import SimpleNamespace

import pytest
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import EvaluationCallback
from skyrl_train.rollouts.buffer import BatchPolicy, RolloutBuffer, RolloutBufferConfig
from skyrl_train.rollouts.context import EvalDrainStrategy, RolloutRequestSpec, TrainingContext
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads
from tests.cpu.task_specs import lowered_task, task_input, text_task


class Dataset:
    def __init__(self, selected):
        self.selected = selected

    def __len__(self):
        return 1

    def uid(self, index):
        return "task"

    def collate_fn(self, rows):
        return rows

    def __getitem__(self, index):
        self.selected.set()
        return {"uid": "task", "input": task_input(lowered_task(text_task(), "agent"))}


@pytest.mark.asyncio
async def test_eval_pause_overlaps_existing_work_and_preserves_lease_crossing_pause(monkeypatch):
    config = RolloutBufferConfig(1, 1, 0, BatchPolicy.FULL_BATCH, None, None)
    buffer = RolloutBuffer(config)
    selected = asyncio.Event()
    submitted = asyncio.Event()
    finish = asyncio.Event()
    dataset = Dataset(selected)
    loader = PromptLoader(dataset, SeededPasses(1, seed=0, shuffle=False), batch_size=1, max_stale_retries=0, max_unusable_groups=4)

    async def run_task(task, writer):
        submitted.set()
        await finish.wait()
        return 0

    async def snapshot():
        return buffer.snapshot()

    async def acquire_lease():
        lease = await buffer.acquire_lease()
        # The actor response races the pause, as it can when acquisition was
        # already pending at the end of training's batch admission.
        await EvaluationCallback(eval_steps=3).on_step_begin_async(
            TrainerState(3, 0, 32, 32), TrainerControl(),
            trainer=SimpleNamespace(eval_dataset=dataset, context=context),
        )
        return lease

    actor = SimpleNamespace(
        acquire_lease=SimpleNamespace(remote=acquire_lease),
        snapshot=SimpleNamespace(remote=snapshot),
        publish=SimpleNamespace(remote=buffer.publish),
    )
    context = TrainingContext(
        loader, config, actor, None, RolloutRequestSpec(1, {}),
        SimpleNamespace(run_task=run_task), MemoryPayloads(), rollout_spans=False, eval_drain=EvalDrainStrategy.WAIT,
    )
    monkeypatch.setattr("skyrl_train.rollouts.context.ray.kill", lambda actor: None)
    context.start()
    try:
        await context.publish(1)
        await asyncio.wait_for(selected.wait(), timeout=5)
        await context.pause_and_drain()
        assert not submitted.is_set()
        state = await context.state_dict()
        assert [row["uid"] for row in state.loader.retries] == ["task"]
        await context.resume()
        await asyncio.wait_for(submitted.wait(), timeout=5)
        finish.set()
        await context.pause_and_drain()
    finally:
        await context.close()
