import asyncio
import pickle

import pytest

from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionStalledError
from skyrl_train.config.rollout_validation import FullBatchFill
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutLease,
    RolloutVerdict,
    BatchFillDeadlineExceeded,
)
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses
from skyrl_train.telemetry import GeneratedWork
from tests.cpu.task_specs import lowered_task, task_input, text_task


def prompt(uid, source, route="nonagentic", family=None):
    return {"uid": uid, "input": task_input(lowered_task(text_task(), "agent"), data_source=source),
            "group_route": route, "family": family}


async def commit_group(buffer, lease, row, rejection=None):
    verdict = RolloutVerdict(
        row["uid"], (rejection,) if rejection is not None else (),
        None if rejection is not None else GroupSelectionResult.KEEP,
        None if rejection is not None else GroupRewards((0., 1.), (0., 1.), True), GeneratedWork(2, 2, 20),
    )
    await buffer.commit(lease.lease_id, row, verdict, PayloadReference(row["uid"] if rejection is None else None))


@pytest.mark.asyncio
@pytest.mark.parametrize("bounded", [False, True])
async def test_straggler_exchange_preserves_both_batches_and_does_not_drop_the_slow_prompt(bounded):
    now = [0.]
    config = RolloutBufferConfig(2, 4, 1, BatchPolicy.FULL_BATCH, None, None,
                               consumed_route_quotas={"nonagentic": 2},
                               full_batch_fill=FullBatchFill(600, 900) if bounded else None)
    buffer = RolloutBuffer(config, clock=lambda: now[0])
    await buffer.publish(1)
    leases = [await buffer.acquire_lease() for _ in range(4)]
    rows = [prompt("a", "math"), prompt("slow", "puzzles"), prompt("b", "math"), prompt("reserve", "puzzles")]
    if bounded:
        for lease, row in zip(leases, rows, strict=True):
            await buffer.bind_prompt(lease.lease_id, row)
    for index in [0, 2, 3]:
        await commit_group(buffer, leases[index], rows[index])
    partial = await buffer.admit(1)
    assert [row.uid for row in partial.admitted] == ["a"]
    assert partial.selection is None
    now[0] = 600
    if not bounded:
        with pytest.raises(GroupAdmissionStalledError):
            await buffer.admit(0)
        await commit_group(buffer, leases[1], rows[1])
    first = await buffer.admit(1)
    assert first.selection is not None
    assert buffer.payload_refs(1, [0, 1]) == ["a", "reserve" if bounded else "slow"]
    if bounded:
        # It still commits under the original worker lease, now exchanged into batch2.
        await commit_group(buffer, leases[1], rows[1])
    await buffer.publish(2)
    second = await buffer.admit(1)
    assert second.selection is not None
    assert set(buffer.payload_refs(2, [0, 1])) == {"b", "slow" if bounded else "reserve"}
    assert first.selection.metrics["async/consumed_route/nonagentic"] == 2
    assert second.selection.metrics["async/consumed_route/nonagentic"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", [AdmissionRejection.FULLY_MASKED, AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE])
async def test_rejected_slot_replacement_keeps_source_even_when_another_source_is_available(rejection):
    config = RolloutBufferConfig(1, 1, 0, BatchPolicy.FULL_BATCH, None, None,
                               consumed_route_quotas={"nonagentic": 1}, full_batch_fill=FullBatchFill(600, 900))
    buffer = RolloutBuffer(config)
    await buffer.publish(1)
    original = prompt("bad", "puzzles")
    lease = await buffer.acquire_lease()
    await buffer.bind_prompt(lease.lease_id, original)
    await commit_group(buffer, lease, original, rejection)
    await buffer.admit(1)
    # The source debt survives restart even though the rejected payload is gone.
    restored = RolloutBuffer(config)
    await restored.restore(buffer.snapshot())
    buffer = restored
    await buffer.publish(1)
    replacement = await buffer.acquire_lease()
    rows = [prompt("other", "math"), original, prompt("good", "puzzles")]

    class Dataset:
        def __len__(self):
            return len(rows)

        def uid(self, index):
            return rows[index]["uid"]

        def __getitem__(self, index):
            return rows[index]

        def collate_fn(self, values):
            return values

    loader = PromptLoader(Dataset(), SeededPasses(3, seed=0, shuffle=False), batch_size=1,
                          max_stale_retries=0, max_unusable_groups=3,
                          routes={row["uid"]: "nonagentic" for row in rows},
                          sources={"other": "math", "bad": "puzzles", "good": "puzzles"})
    row = loader.next_prompt({"bad"}, replacement.route, replacement.family, replacement.stratum.source)
    assert row["uid"] == "good"
    await buffer.bind_prompt(replacement.lease_id, row)
    await commit_group(buffer, replacement, row)
    admission = await buffer.admit(1)
    assert admission.selection is not None
    assert buffer.payload_refs(1, [0]) == ["good"]


@pytest.mark.asyncio
async def test_batch_deadline_does_not_borrow_another_source_or_reset_on_partial_progress():
    now = [0.]
    config = RolloutBufferConfig(2, 4, 1, BatchPolicy.FULL_BATCH, None, None,
                               consumed_route_quotas={"nonagentic": 2}, full_batch_fill=FullBatchFill(600, 900))
    buffer = RolloutBuffer(config, clock=lambda: now[0])
    await buffer.publish(1)
    leases = [await buffer.acquire_lease() for _ in range(4)]
    rows = [prompt("ready", "math"), prompt("slow", "puzzles"), prompt("b", "math"), prompt("c", "math")]
    for lease, row in zip(leases, rows, strict=True):
        await buffer.bind_prompt(lease.lease_id, row)
    for index in [0, 2, 3]:
        await commit_group(buffer, leases[index], rows[index])
    partial = await buffer.admit(1)
    assert partial.selection is None
    now[0] = 600
    with pytest.raises(GroupAdmissionStalledError):
        await buffer.admit(0)
    now[0] = 901
    with pytest.raises(BatchFillDeadlineExceeded, match="missing_groups=1"):
        await buffer.admit(10000)


@pytest.mark.asyncio
async def test_straggler_exchange_cannot_move_old_policy_work_beyond_its_staleness_window():
    now = [0.]
    config = RolloutBufferConfig(1, 2, 1, BatchPolicy.FULL_BATCH, None, None,
                               consumed_route_quotas={"nonagentic": 1}, full_batch_fill=FullBatchFill(600, 900))
    buffer = RolloutBuffer(config, clock=lambda: now[0])
    await buffer.publish(1)
    first, slow = [await buffer.acquire_lease() for _ in range(2)]
    for lease, row in [(first, prompt("first", "math")), (slow, prompt("slow", "math"))]:
        await buffer.bind_prompt(lease.lease_id, row)
    await commit_group(buffer, first, prompt("first", "math"))
    assert (await buffer.admit(1)).selection is not None
    await buffer.publish(2)
    reserve = await buffer.acquire_lease()
    await buffer.bind_prompt(reserve.lease_id, prompt("reserve", "math"))
    await commit_group(buffer, reserve, prompt("reserve", "math"))
    with pytest.raises(GroupAdmissionStalledError):
        await buffer.admit(0)
    now[0] = 600
    with pytest.raises(GroupAdmissionStalledError):
        await buffer.admit(0)
    await commit_group(buffer, slow, prompt("slow", "math"))
    assert (await buffer.admit(1)).selection is not None
    assert buffer.payload_refs(2, [0]) == ["slow"]
    await buffer.publish(3)
    assert (await buffer.admit(1)).selection is not None
    assert buffer.payload_refs(3, [0]) == ["reserve"]
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
async def test_rolling_route_quotas_take_first_committed_per_route():
    config = RolloutBufferConfig(3, 9, 2, BatchPolicy.ROLLING, None, None,
                                consumed_route_quotas={"agentic": 1, "nonagentic": 2})
    buffer = RolloutBuffer(config)
    await buffer.publish(1)
    leases = [await buffer.acquire_lease() for _ in range(9)]
    a = [lease for lease in leases if lease.route == "agentic"]
    n = [lease for lease in leases if lease.route == "nonagentic"]
    for lease, uid in [(a[2], "a-first"), (a[1], "a-second"), (n[-1], "n-first"), (n[-2], "n-second")]:
        await commit_group(buffer, lease, prompt(uid, "source", lease.route))
    first = await buffer.admit(1)
    assert first.selection is not None
    assert buffer.payload_refs(1, [0, 1, 2]) == ["a-first", "n-first", "n-second"]
    assert first.selection.metrics["async/consumed_route/agentic"] == 1
    assert first.selection.metrics["async/consumed_route/nonagentic"] == 2
    await buffer.publish(2)
    for lease, uid in [(n[-3], "n-third"), (n[-4], "n-fourth")]:
        await commit_group(buffer, lease, prompt(uid, "source", lease.route))
    second = await buffer.admit(1)
    assert second.selection is not None
    assert buffer.payload_refs(2, [0, 1, 2]) == ["a-second", "n-third", "n-fourth"]


@pytest.mark.asyncio
async def test_rolling_route_quotas_discard_stale_and_return_prompts():
    buffer = RolloutBuffer(RolloutBufferConfig(2, 4, 1, BatchPolicy.ROLLING, None, None,
                                             consumed_route_quotas={"agentic": 1, "nonagentic": 1}))
    await buffer.publish(1)
    slow_a, slow_n, fast_a, fast_n = [await buffer.acquire_lease() for _ in range(4)]
    for lease in [fast_a, fast_n]:
        await commit_group(buffer, lease, prompt(lease.lease_id, "fast", lease.route))
    assert (await buffer.admit(1)).selection is not None
    await buffer.publish(2)
    for _ in range(2):
        lease = await buffer.acquire_lease()
        await commit_group(buffer, lease, prompt(lease.lease_id, "fast", lease.route))
    assert (await buffer.admit(1)).selection is not None
    await buffer.publish(3)
    slow = [prompt("slow-a", "slow", "agentic"), prompt("slow-n", "slow", "nonagentic")]
    for lease, row in zip([slow_a, slow_n], slow, strict=True):
        await commit_group(buffer, lease, row)
    admission = await buffer.admit(1)
    assert admission.retries == slow
    assert [(row.uid, row.disposition, row.source) for row in admission.dispositions] == [
        ("slow-a", AdmissionRejection.STALE.value, "slow"), ("slow-n", AdmissionRejection.STALE.value, "slow"),
    ]
    assert admission.admitted == []
    assert admission.selection is None


@pytest.mark.asyncio
async def test_rolling_checkpoint_resume_preserves_route_commit_order():
    config = RolloutBufferConfig(2, 4, 1, BatchPolicy.ROLLING, None, None,
                                consumed_route_quotas={"agentic": 1, "nonagentic": 1})
    buffer = RolloutBuffer(config)
    await buffer.publish(1)
    leases = [await buffer.acquire_lease() for _ in range(4)]
    a = [lease for lease in leases if lease.route == "agentic"]
    for lease, uid in zip(reversed(a), ["a-first", "a-second"], strict=True):
        await commit_group(buffer, lease, prompt(uid, "source", "agentic"))
    snapshot = pickle.loads(pickle.dumps(buffer.snapshot()))
    resumed = RolloutBuffer(config)
    await resumed.restore(snapshot)
    await resumed.publish(1)
    for step in [1, 2]:
        if step == 2:
            await resumed.publish(step)
        lease = await resumed.acquire_lease()
        if lease.route != "nonagentic":
            lease = await resumed.acquire_lease()
        assert lease.route == "nonagentic"
        await commit_group(resumed, lease, prompt(f"n-{step}", "source", "nonagentic"))
        assert (await resumed.admit(1)).selection is not None
        assert resumed.payload_refs(step, [0, 1]) == ["a-first" if step == 1 else "a-second", f"n-{step}"]


@pytest.mark.asyncio
async def test_rolling_resume_refuses_full_batch_checkpoint_even_when_empty():
    full = RolloutBuffer(RolloutBufferConfig(2, 4, 1, BatchPolicy.FULL_BATCH, None, None,
                                           consumed_route_quotas={"agentic": 1, "nonagentic": 1}))
    snapshot = pickle.loads(pickle.dumps(full.snapshot()))
    rolling = RolloutBuffer(RolloutBufferConfig(2, 4, 1, BatchPolicy.ROLLING, None, None,
                                              consumed_route_quotas={"agentic": 1, "nonagentic": 1}))
    with pytest.raises(ValueError, match="Cannot resume rolling batch policy from full_batch checkpoint"):
        await rolling.restore(snapshot)
