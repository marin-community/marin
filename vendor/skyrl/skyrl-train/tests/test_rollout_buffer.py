import asyncio

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.buffer import (
    SAMPLING_SOURCE,
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
            max_in_flight=4,
            max_staleness_steps=1,
            batch_policy=BatchPolicy.FULL_BATCH,
            dynamic_sampling=None,
            max_candidate_groups=None,
            groups_per_family={"agentic": 1, "math": 1},
        )
    )

    async def commit(lease: RolloutLease, *, rejected: bool = False, family: str | None = None) -> None:
        verdict = RolloutVerdict(
            lease.lease_id,
            (AdmissionRejection.FULLY_MASKED,) if rejected else (),
            None if rejected else GroupSelectionResult.KEEP,
            None if rejected else GroupRewards((0.0, 1.0), (0.0, 1.0), True),
            GeneratedWork(2, 2, 20),
        )
        payload = PayloadReference(None if rejected else lease.lease_id)
        await buffer.commit(lease.lease_id, {"family": family if family is not None else lease.family}, verdict, payload)

    async with asyncio.timeout(5):
        await buffer.publish(1)
        first, rejected, *later = [await buffer.acquire_lease() for _ in range(4)]
        assert (first.family, rejected.family) == ("agentic", "math")
        with pytest.raises(ValueError, match="family does not match its rollout lease"):
            await commit(first, family=rejected.family)
        others = [first, *later]
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
            assert set(buffer.payload_refs(step, [rollout.index for rollout in admission.admitted])) == (
                {first.lease_id, replacement.lease_id} if step == 1 else {lease.lease_id for lease in later}
            )


@pytest.mark.asyncio
async def test_checkpoint_preserves_generated_cost_of_pending_sampled_rejections():
    config = RolloutBufferConfig(
        batch_size=1,
        max_in_flight=1,
        max_staleness_steps=0,
        batch_policy=BatchPolicy.FULL_BATCH,
        dynamic_sampling=None,
        max_candidate_groups=None,
    )
    buffer = RolloutBuffer(config)
    await buffer.publish(1)
    lease = await buffer.acquire_lease()
    rejected = RolloutVerdict(
        "8",
        (AdmissionRejection.FULLY_MASKED,),
        None,
        None,
        GeneratedWork(2, 2, 27),
        rejected_verifier_scores=(None, None),
    )
    await buffer.commit(lease.lease_id, {SAMPLING_SOURCE: "fixture"}, rejected, PayloadReference(None))

    restored = RolloutBuffer(config)
    await restored.restore(buffer.snapshot())
    await restored.publish(1)
    admission = await restored.admit(1)
    (outcome,) = admission.dispositions
    assert outcome.uid == "8"
    assert outcome.tokens == 27
    assert outcome.sampling_result is not None
    assert outcome.sampling_result.lease_id == lease.lease_id
    assert outcome.sampling_result.dispatch_step == 1
    assert outcome.sampling_result.verified_scores == (None, None)
