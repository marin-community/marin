import asyncio

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutLease,
    RolloutVerdict,
)
from skyrl_train.telemetry import GeneratedWork


@pytest.mark.asyncio
async def test_rejected_rollout_reopens_the_same_family_slot():
    buffer = RolloutBuffer(
        RolloutBufferConfig(
            batch_size=2,
            max_in_flight=4,
            max_staleness_steps=1,
            batch_policy=BatchPolicy.FULL_BATCH,
            dynamic_sampling=None,
            max_candidate_groups=None,
            family_quotas={"agentic": 1, "math": 1},
        )
    )

    async def commit(lease: RolloutLease, *, rejected: bool = False) -> None:
        verdict = RolloutVerdict(
            lease.lease_id,
            (AdmissionRejection.FULLY_MASKED,) if rejected else (),
            None if rejected else GroupSelectionResult.KEEP,
            None if rejected else GroupRewards((0.0, 1.0), (0.0, 1.0), True),
            GeneratedWork(2, 2, 20),
        )
        await buffer.commit(lease.lease_id, {"family": lease.family}, verdict, PayloadReference(lease.lease_id))

    async with asyncio.timeout(5):
        await buffer.publish(1)
        rejected, *others = [await buffer.acquire_lease() for _ in range(4)]
        for lease in reversed(others):
            await commit(lease)
        await commit(rejected, rejected=True)
        replacement = await buffer.acquire_lease()
        assert (replacement.batch_id, replacement.family) == (1, rejected.family)
        await commit(replacement)

        for step in (1, 2):
            if step > 1:
                await buffer.publish(step)
            admission = await buffer.admit(1)
            assert admission.selection is not None
            assert admission.selection.metrics["async/consumed_family/agentic"] == 1
            assert admission.selection.metrics["async/consumed_family/math"] == 1
