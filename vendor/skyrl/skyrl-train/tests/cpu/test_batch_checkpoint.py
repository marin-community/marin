import asyncio

import numpy as np
import pytest
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import BatchPolicy, RolloutBufferConfig, RolloutContentPolicy, RolloutGroup
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext, start_rollout_buffer
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads

from tests.cpu.task_specs import lowered_task, task_input, text_task

pytestmark = [pytest.mark.slow, pytest.mark.timeout(180)]


class Prompts:
    def __init__(self):
        self.uids = ["a", "b", "c"]

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        return self.uids[index]

    def uid(self, index):
        return self.uids[index]

    def collate_fn(self, uids):
        return [{"uid": uid, "input": task_input(lowered_task(text_task(uid)))} for uid in uids]


class Rollouts:
    def __init__(self, *, hold_b=False):
        self.hold_b = hold_b
        self.held = asyncio.Event()
        self.c_written = asyncio.Event()
        self.started = []

    async def run_task(self, task, writer):
        uid = task.prompt["uid"]
        self.started.append(uid)
        if self.hold_b and uid == "b":
            await self.held.wait()
        width = ord(uid) - ord("a") + 1
        batch = {
            "prompt_token_ids": [[1], [1]],
            "response_ids": [[2] * width, [3] * (width + 1)],
            "rewards": [0.0, float(width)],
            "loss_masks": [[1] * width, [1] * (width + 1)],
            "rollout_logprobs": None,
        }
        await writer.write_rollout(task.lease, RolloutGroup(batch, uid, task.lease.policy_step, task.prompt))
        if uid == "c":
            self.c_written.set()
        return sum(map(len, batch["response_ids"]))


def context_for(workers, payloads):
    config = RolloutBufferConfig(1, 2, 1, BatchPolicy.FULL_BATCH, None, None)
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(None),
    )
    return TrainingContext(
        PromptLoader(Prompts(), SeededPasses(3, seed=0, shuffle=False), batch_size=1),
        config,
        start_rollout_buffer(config),
        policy,
        RolloutRequestSpec(samples_per_prompt=2, sampling_params={}),
        workers,
        payloads,
        rollout_spans=False,
    )


async def ignore_admitted(_groups):
    pass


async def next_groups(context):
    groups, _ = await context.next_batch(stall_timeout=30, on_admitted=ignore_admitted)
    return groups


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_kind", ["memory", "object_store"])
async def test_checkpoint_captures_committed_payload_and_row_facts(ray_module, tmp_path, payload_kind):
    payloads = MemoryPayloads() if payload_kind == "memory" else ObjectStorePayloads(str(tmp_path / "objects"))
    workers = Rollouts(hold_b=True)
    original = context_for(workers, payloads)
    original.start()
    try:
        await original.publish(1)
        assert [group.uid for group in await next_groups(original)] == ["a"]
        await original.publish(2)
        await asyncio.wait_for(workers.c_written.wait(), timeout=30)
        state = await original.state_dict()
    finally:
        await original.close()
    assert [group.verdict.uid for group in state.ready] == ["c"]
    saved = state.ready[0]
    payload = (await payloads.fetch([payloads.restore(saved.payload)]))[0].trajectory_batch
    facts = saved.verdict.row_facts
    np.testing.assert_array_equal(facts.prompt_len, [len(row) for row in payload["prompt_token_ids"]])
    np.testing.assert_array_equal(facts.response_len, [len(row) for row in payload["response_ids"]])
    np.testing.assert_array_equal(facts.compensated_metric_totals, payload["rewards"])
    assert facts.rewards.optimization == (0.0, 3.0)
    assert facts.response_len.tolist() == [3, 4]
    assert facts.rollout_passes == (False, True)
