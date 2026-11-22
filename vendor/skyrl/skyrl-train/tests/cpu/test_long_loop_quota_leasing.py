import asyncio

from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import BatchPolicy, PayloadReference, RolloutBuffer, RolloutBufferConfig, RolloutVerdict, RolloutContentPolicy, RolloutGroup
from skyrl_train.telemetry import GeneratedWork


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
