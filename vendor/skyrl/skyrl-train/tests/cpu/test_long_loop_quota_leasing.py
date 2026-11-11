import asyncio
from collections import Counter

from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.buffer import BatchPolicy, PayloadReference, RolloutBuffer, RolloutBufferConfig, RolloutVerdict
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
