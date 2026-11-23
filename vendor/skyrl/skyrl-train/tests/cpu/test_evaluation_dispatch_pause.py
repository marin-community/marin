"""Evaluation admission pauses preserve pending prompts in restart state."""

import asyncio
import pickle
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


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_policy", [BatchPolicy.FULL_BATCH, BatchPolicy.ROLLING])
async def test_checkpoint_preserves_batch_policy_and_refuses_full_batch_as_rolling(saved_policy):
    def make(policy):
        config = RolloutBufferConfig(1, 2, 1, policy, None, None,
                                     consumed_route_quotas={"nonagentic": 1})
        buffer = RolloutBuffer(config)

        async def snapshot():
            return buffer.snapshot()

        actor = SimpleNamespace(snapshot=SimpleNamespace(remote=snapshot), restore=SimpleNamespace(remote=buffer.restore))
        dataset = Dataset(asyncio.Event())
        loader = PromptLoader(dataset, SeededPasses(1, seed=0, shuffle=False), batch_size=1,
                              max_stale_retries=0, max_unusable_groups=4)
        return TrainingContext(loader, config, actor, None, RolloutRequestSpec(1, {}),
                               SimpleNamespace(), MemoryPayloads(), rollout_spans=False,
                               eval_drain=EvalDrainStrategy.PRIORITY)

    original = make(saved_policy)
    original.loader.next_prompt(set())
    state = pickle.loads(pickle.dumps(await original.state_dict()))
    resumed = make(BatchPolicy.ROLLING)
    if saved_policy is BatchPolicy.FULL_BATCH:
        with pytest.raises(ValueError, match="Cannot resume rolling batch policy from full_batch checkpoint"):
            await resumed.load_state_dict(state)
    else:
        await resumed.load_state_dict(state)
        assert resumed.loader.state_dict() == state.loader
        assert (await resumed.state_dict()).batch_policy is BatchPolicy.ROLLING
