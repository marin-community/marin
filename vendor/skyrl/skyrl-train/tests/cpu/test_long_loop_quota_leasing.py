import asyncio
from collections import Counter

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import BatchPolicy, FullBatchRolloutPolicy, GroupRewards, PayloadReference, RolloutBuffer, RolloutBufferConfig, RolloutVerdict, RolloutContentPolicy, RolloutGroup
from skyrl_train.telemetry import GeneratedWork


def test_rejected_agentic_groups_do_not_starve_nonagentic_leases():
    async def exercise():
        buffer = RolloutBuffer(RolloutBufferConfig(64, 32, 2, BatchPolicy.FULL_BATCH, None, None, {"agentic": 32, "non_agentic": 32}))
        await buffer.publish(1)
        leases = [await buffer.acquire_lease() for _ in range(32)]
        assert Counter(lease.route for lease in leases) == {"agentic": 16, "non_agentic": 16}
        agentic = next(lease for lease in leases if lease.route == "agentic")
        await buffer.commit(agentic.lease_id, {"uid": "failed", "group_route": "agentic"},
                            RolloutVerdict("failed", (AdmissionRejection.FULLY_MASKED,), None, None, GeneratedWork(8, 8, 0)),
                            PayloadReference(None))
        replacement = await buffer.acquire_lease()
        await buffer.commit(replacement.lease_id, {"uid": "failed-again", "group_route": replacement.route},
                            RolloutVerdict("failed-again", (AdmissionRejection.FULLY_MASKED,), None, None, GeneratedWork(8, 8, 0)),
                            PayloadReference(None))
        next_lease = await buffer.acquire_lease()
        assert {replacement.route, next_lease.route} == {"agentic", "non_agentic"}
    asyncio.run(exercise())


@pytest.mark.parametrize("samples_per_prompt", [4, 8])
def test_generation_boundary_prevents_prefetch_beyond_eight_batches(samples_per_prompt):
    async def exercise():
        groups = 128 // samples_per_prompt
        config = RolloutBufferConfig(groups, groups * 3, 2, BatchPolicy.FULL_BATCH, None, None,
                                     {"agentic": groups // 2, "nonagentic": groups // 2},
                                     generation_batch_limit=8)
        buffer = RolloutBuffer(config)
        await buffer.publish(1)
        generated = Counter()
        issued = Counter()
        for step in range(1, 9):
            # Fill the whole staleness window, including future batches, before consuming.
            for batch_id in range(step, min(step + 2, 8) + 1):
                for _ in range(groups - issued[batch_id]):
                    lease = await buffer.acquire_lease()
                    assert lease.batch_id == batch_id
                    uid = f"{batch_id}-{issued[batch_id]}"
                    issued[batch_id] += 1
                    generated[lease.route] += samples_per_prompt
                    rewards = GroupRewards((0.0,) * samples_per_prompt, (0.0,) * samples_per_prompt, False)
                    verdict = RolloutVerdict(uid, (), GroupSelectionResult.KEEP, rewards,
                                             GeneratedWork(samples_per_prompt, samples_per_prompt, 0))
                    await buffer.commit(lease.lease_id, {"uid": uid, "group_route": lease.route}, verdict,
                                        PayloadReference(uid))
            admission = await buffer.admit(timeout=1)
            assert admission.selection is not None
            assert sum(group.sample_count for group in admission.admitted) == 128
            await buffer.publish(step + 1)
        assert generated == {"agentic": 512, "nonagentic": 512}
        policy = FullBatchRolloutPolicy(config)
        assert policy.lease_batch(7, Counter({7: groups, 8: groups})) is None
        assert policy.lease_batch(9, Counter()) is None
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(buffer.acquire_lease(), timeout=0.01)
    asyncio.run(exercise())


def test_fixed_generation_budget_fails_instead_of_replacing_a_masked_group():
    async def exercise():
        buffer = RolloutBuffer(RolloutBufferConfig(1, 1, 2, BatchPolicy.FULL_BATCH, None, None,
                                                   generation_batch_limit=8))
        await buffer.publish(1)
        lease = await buffer.acquire_lease()
        verdict = RolloutVerdict("masked", (AdmissionRejection.FULLY_MASKED,), None, None, GeneratedWork(8, 8, 0))
        with pytest.raises(RuntimeError, match="Fixed generation budget cannot replace group masked: fully_masked"):
            await buffer.commit(lease.lease_id, {"uid": "masked"}, verdict, PayloadReference(None))
    asyncio.run(exercise())


def test_masked_group_preserves_exception_counts_for_admission_logs():
    async def exercise():
        content = RolloutContentPolicy(
            GroupAdmissionPolicy(GroupAdvantageInvariant.exact_physical(physical_group_size=2),
                                 rollout_logprobs_required=False),
            GroupSelectionPolicy(None),
        )
        group = RolloutGroup({"response_ids": [[21], [22]], "rewards": [0.0, 0.0],
                              "loss_masks": [[0], [0]],
                              "exception_types": ["NonZeroAgentExitCodeError", "VerifierRuntimeError"]},
                             "failed-group", 1, {})
        buffer = RolloutBuffer(RolloutBufferConfig(1, 1, 2, BatchPolicy.FULL_BATCH, None, None))
        await buffer.publish(1)
        lease = await buffer.acquire_lease()
        await buffer.commit(lease.lease_id, {}, content.verdict(group), PayloadReference(None))
        progress = await buffer.admit(timeout=1)
        assert progress.dispositions[0].disposition == "fully_masked"
        assert dict(progress.dispositions[0].masked_reasons) == {
            "NonZeroAgentExitCodeError": 1, "VerifierRuntimeError": 1,
        }
    asyncio.run(exercise())
