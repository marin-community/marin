"""Asynchronous failures must not change the source mixture the learner consumes."""

import collections

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.rollouts.buffer import (
    BatchPolicy, GroupRewards, PayloadReference, RolloutBuffer, RolloutBufferConfig, RolloutVerdict,
)
from skyrl_train.telemetry import GeneratedWork
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses


@pytest.mark.asyncio
async def test_out_of_order_and_rejected_groups_keep_exact_consumed_quota():
    buffer = RolloutBuffer(RolloutBufferConfig(
        batch_size=4, max_in_flight=8, max_staleness_steps=2, batch_policy=BatchPolicy.FULL_BATCH,
        dynamic_sampling=None, max_candidate_groups=None, consumed_route_quotas={'agentic': 2, 'nonagentic': 2},
    ))
    await buffer.publish(1)
    leases = [await buffer.acquire_lease() for _ in range(8)]
    assert collections.Counter(lease.route for lease in leases[:4]) == {'agentic': 2, 'nonagentic': 2}

    async def commit(lease, selection=GroupSelectionResult.KEEP):
        prompt = {'uid': lease.lease_id, 'group_route': lease.route}
        verdict = RolloutVerdict(lease.lease_id, (), selection, GroupRewards((0., 1.), (0., 1.), True), GeneratedWork(2, 2, 20))
        await buffer.commit(lease.lease_id, prompt, verdict, PayloadReference(lease.lease_id))

    # Fast future groups and non-agentic groups arrive first; one agentic group is discarded.
    for lease in reversed(leases[1:]):
        await commit(lease)
    await commit(leases[0], GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD)
    replacement = await buffer.acquire_lease()
    assert replacement.batch_id == 1 and replacement.route == 'agentic'
    await commit(replacement)
    result = await buffer.admit(1)
    assert result.selection.metrics['async/consumed_route/agentic'] == 2
    assert result.selection.metrics['async/consumed_route/nonagentic'] == 2
    await buffer.publish(2)
    second = await buffer.admit(1)
    assert second.selection.metrics['async/consumed_route/agentic'] == 2
    assert second.selection.metrics['async/consumed_route/nonagentic'] == 2


@pytest.mark.asyncio
async def test_wrong_route_cannot_consume_another_pools_reserved_slot():
    buffer = RolloutBuffer(RolloutBufferConfig(
        batch_size=2, max_in_flight=2, max_staleness_steps=2, batch_policy=BatchPolicy.FULL_BATCH,
        dynamic_sampling=None, max_candidate_groups=None, consumed_route_quotas={'a': 1, 'b': 1},
    ))
    await buffer.publish(1)
    lease = await buffer.acquire_lease()
    verdict = RolloutVerdict('row', (), GroupSelectionResult.KEEP, GroupRewards((1.,), (1.,), True), GeneratedWork(1, 1, 1))
    with pytest.raises(ValueError, match='consumed-route lease'):
        await buffer.commit(lease.lease_id, {'uid': 'row', 'group_route': 'b'}, verdict, PayloadReference('row'))


def test_loader_routes_retries_and_fallback_without_crossing_pools():
    class Rows:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

        def uid(self, index):
            return str(index)

        def collate_fn(self, rows):
            return [{'uid': str(index)} for index in rows]

    loader = PromptLoader(Rows(), SeededPasses(4, seed=4, shuffle=True), batch_size=2,
                          max_stale_retries=1, routes={'0': 'a', '1': 'a', '2': 'b', '3': 'b'})
    first = loader.next_prompt(set(), 'a')
    loader.retry(first)
    other = loader.next_prompt(set(), 'b')
    assert other['group_route'] == 'b'
    assert loader.next_prompt(set(), 'a') == first
    assert loader.next_prompt({'0', '1'}, 'a') is None
