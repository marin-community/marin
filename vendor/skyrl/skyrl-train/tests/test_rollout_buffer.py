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
async def test_family_leases_fill_batches_reject_wrong_family_and_reopen_slots():
    buffer = RolloutBuffer(
        RolloutBufferConfig(
            batch_size=2,
            max_outstanding_leases=4,
            max_staleness_steps=1,
            batch_policy=BatchPolicy.FULL_BATCH,
            dynamic_sampling=None,
            max_candidate_groups=None,
            groups_per_family={"agentic": 1, "math": 1},
        )
    )

    async def submit_group(lease: RolloutLease, *, rejected: bool = False, family: str | None = None) -> None:
        verdict = RolloutVerdict(
            lease.lease_id,
            (AdmissionRejection.FULLY_MASKED,) if rejected else (),
            None if rejected else GroupSelectionResult.KEEP,
            None if rejected else GroupRewards((0.0, 1.0), (0.0, 1.0), True),
            GeneratedWork(2, 2, 20),
        )
        payload = PayloadReference(None if rejected else lease.lease_id)
        entry = {"family": family if family is not None else lease.family}
        await buffer.submit_group(lease.lease_id, entry=entry, verdict=verdict, payload=payload)

    async with asyncio.timeout(5):
        await buffer.publish(1)
        first, rejected, *later = [await buffer.acquire_lease() for _ in range(4)]
        assert (first.family, rejected.family) == ("agentic", "math")
        with pytest.raises(ValueError, match="family does not match its rollout lease"):
            await submit_group(first, family=rejected.family)
        others = [first, *later]
        for lease in reversed(others):
            await submit_group(lease)
        await submit_group(rejected, rejected=True)
        replacement = await buffer.acquire_lease()
        assert (replacement.batch_id, replacement.family) == (1, rejected.family)
        await submit_group(replacement)

        for step in (1, 2):
            if step > 1:
                await buffer.publish(step)
            admission = await buffer.admit(1)
            assert admission.selection is not None
            assert set(buffer.payload_refs(step, [rollout.index for rollout in admission.admitted])) == (
                {first.lease_id, replacement.lease_id} if step == 1 else {lease.lease_id for lease in later}
            )
