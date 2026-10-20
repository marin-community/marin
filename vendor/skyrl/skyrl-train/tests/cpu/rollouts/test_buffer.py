"""Lease accounting and batch selection of the rollout buffer, exercised in-process without Ray."""

import asyncio

import pytest
from skyrl_gym.verification import VerificationResult

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    GroupAdmissionStalledError,
)
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutBufferConfig,
    RolloutVerdict,
)
from skyrl_train.rollouts.loader import JudgedGroup
from skyrl_train.telemetry import GeneratedWork

# Long enough that an expected admission never times out, short enough that a blocked lease fails fast.
PROGRESS_TIMEOUT = 5.0
BLOCKED_TIMEOUT = 0.05
UNIFORM_REWARDS = GroupRewards(optimization=(0.0, 0.0), outcome=(0.0, 0.0), passed=False)
SPREAD_REWARDS = GroupRewards(optimization=(0.0, 1.0), outcome=(0.0, 1.0), passed=True)


@pytest.fixture(params=list(BatchPolicy))
def batch_policy(request) -> BatchPolicy:
    return request.param


def _buffer(
    batch_policy: BatchPolicy,
    *,
    batch_size: int,
    max_in_flight: int = 4,
    max_staleness_steps: int = 1,
    dynamic_sampling: DynamicSamplingType | None = None,
    max_candidate_groups: int | None = None,
) -> RolloutBuffer:
    return RolloutBuffer(
        RolloutBufferConfig(
            batch_size, max_in_flight, max_staleness_steps, batch_policy, dynamic_sampling, max_candidate_groups
        )
    )


def _verdict(
    uid: str,
    *,
    rejection: AdmissionRejection | None = None,
    selection: GroupSelectionResult = GroupSelectionResult.KEEP,
    rewards: GroupRewards = SPREAD_REWARDS,
) -> RolloutVerdict:
    if rejection is not None:
        return RolloutVerdict(uid, (rejection,), None, None, GeneratedWork(2, 2, 8))
    return RolloutVerdict(uid, (), selection, rewards, GeneratedWork(2, 2, 8))


async def _commit(buffer: RolloutBuffer, lease_id: str, uid: str, **verdict) -> None:
    """Commit a group whose payload stands in for its object reference with its UID."""
    await buffer.commit(lease_id, {"uid": uid}, _verdict(uid, **verdict), PayloadReference(uid))


async def _generate(buffer: RolloutBuffer, uid: str, **verdict) -> None:
    lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await _commit(buffer, lease.lease_id, uid, **verdict)


async def _take_batch(buffer: RolloutBuffer) -> tuple[list[str], dict[str, float]]:
    """Admit until the batch completes, resolving each admitted index in order."""
    payloads = []
    while True:
        admission = await buffer.admit(PROGRESS_TIMEOUT)
        payloads.extend(buffer.payload_refs(admission.batch_id, [group.index for group in admission.admitted]))
        if admission.selection is not None:
            return payloads, admission.selection.metrics


async def _lease_is_blocked(buffer: RolloutBuffer) -> bool:
    try:
        await asyncio.wait_for(buffer.acquire_lease(), BLOCKED_TIMEOUT)
    except TimeoutError:
        return True
    return False


@pytest.mark.asyncio
async def test_on_policy_leases_one_batch_per_published_step(batch_policy):
    buffer = _buffer(batch_policy, batch_size=2, max_staleness_steps=0)
    assert await _lease_is_blocked(buffer)

    await buffer.publish(1)
    await _generate(buffer, "a")
    await _generate(buffer, "b")
    assert await _lease_is_blocked(buffer)
    assert (await _take_batch(buffer))[0] == ["a", "b"]
    assert await _lease_is_blocked(buffer)

    await buffer.publish(2)
    lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert lease.policy_step == 2


@pytest.mark.asyncio
async def test_off_policy_generation_runs_ahead_by_the_staleness_bound(batch_policy):
    buffer = _buffer(batch_policy, batch_size=2, max_in_flight=8, max_staleness_steps=1)
    await buffer.publish(1)
    for uid in "abcd":
        await _generate(buffer, uid)
    assert await _lease_is_blocked(buffer)

    await _take_batch(buffer)
    assert await _lease_is_blocked(buffer)
    await buffer.publish(2)
    for _ in range(2):
        await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert await _lease_is_blocked(buffer)


@pytest.mark.asyncio
async def test_leases_never_exceed_max_in_flight(batch_policy):
    buffer = _buffer(batch_policy, batch_size=1, max_in_flight=2, max_staleness_steps=3)
    await buffer.publish(1)
    lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert await _lease_is_blocked(buffer)

    await _commit(buffer, lease.lease_id, "a")
    assert not await _lease_is_blocked(buffer)


@pytest.mark.asyncio
async def test_surplus_groups_wait_for_the_next_step(batch_policy):
    buffer = _buffer(batch_policy, batch_size=1)
    await buffer.publish(1)
    await _generate(buffer, "a")
    await _generate(buffer, "b")
    assert (await _take_batch(buffer))[0] == ["a"]

    await buffer.publish(2)
    assert (await _take_batch(buffer))[0] == ["b"]


@pytest.mark.asyncio
async def test_groups_stream_to_the_trainer_before_the_batch_completes(batch_policy):
    buffer = _buffer(batch_policy, batch_size=2)
    await buffer.publish(1)
    await _generate(buffer, "a")
    first = await buffer.admit(PROGRESS_TIMEOUT)
    assert ([group.index for group in first.admitted], first.selection) == ([0], None)
    assert buffer.payload_refs(first.batch_id, [0]) == ["a"]

    await _generate(buffer, "b")
    second = await buffer.admit(PROGRESS_TIMEOUT)
    assert [group.index for group in second.admitted] == [1]
    assert buffer.payload_refs(second.batch_id, [1, 0]) == ["b", "a"]
    assert second.selection is not None


@pytest.mark.asyncio
async def test_stale_group_returns_its_prompt_for_regeneration():
    buffer = _buffer(BatchPolicy.ROLLING, batch_size=1)
    await buffer.publish(1)
    stale = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    for step, uid in enumerate(["a", "b"], start=2):
        await _generate(buffer, uid)
        assert (await _take_batch(buffer))[0] == [uid]
        await buffer.publish(step)

    await _commit(buffer, stale.lease_id, "stale")
    admission = await buffer.admit(PROGRESS_TIMEOUT)
    assert admission.admitted == []
    assert admission.retries == [{"uid": "stale"}]

    await _generate(buffer, "fresh")
    payloads, metrics = await _take_batch(buffer)
    assert payloads == ["fresh"]
    assert metrics["async/rejected_count/stale"] == 1


@pytest.mark.asyncio
async def test_rejected_and_duplicate_groups_are_dropped_and_counted(batch_policy):
    buffer = _buffer(batch_policy, batch_size=2)
    await buffer.publish(1)
    await _generate(buffer, "masked", rejection=AdmissionRejection.FULLY_MASKED)
    await _generate(buffer, "a")
    await _generate(buffer, "a")
    await _generate(buffer, "b")

    payloads, metrics = await _take_batch(buffer)
    assert payloads == ["a", "b"]
    assert metrics["async/rejected_count"] == 2
    assert metrics["async/rejected_count/fully_masked"] == 1
    assert metrics["async/rejected_count/duplicate_uid"] == 1
    assert metrics["async/rejected_rate"] == 0.5


@pytest.mark.asyncio
async def test_dynamic_sampling_filter_discards_uninformative_groups(batch_policy):
    buffer = _buffer(batch_policy, batch_size=1, dynamic_sampling=DynamicSamplingType.FILTER)
    await buffer.publish(1)
    uniform = {"selection": GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, "rewards": UNIFORM_REWARDS}
    await _generate(buffer, "uniform", **uniform)
    await _generate(buffer, "informative")

    payloads, metrics = await _take_batch(buffer)
    assert payloads == ["informative"]
    assert metrics["async/dynamic_sampling/candidate_count"] == 2
    assert metrics["async/dynamic_sampling/discarded_count"] == 1
    assert metrics["async/dynamic_sampling/candidate_outcome_reward_mean"] == 0.25
    assert metrics["async/dynamic_sampling/candidate_pass_at_2"] == 0.5


@pytest.mark.parametrize(
    ("verdicts", "passed"),
    [
        ((False, False), False),
        ((False, True), True),
    ],
)
def test_group_passes_by_verifier_verdict_rather_than_partial_credit(verdicts, passed):
    rewards = [0.3, 0.8]
    batch = {
        "response_ids": [[1], [2]],
        "rewards": rewards,
        "verification_results": [
            VerificationResult.verified(reward, passed=verdict) for reward, verdict in zip(rewards, verdicts)
        ],
    }

    assert GroupRewards.from_batch(batch).passed is passed


@pytest.mark.asyncio
async def test_batch_selection_reports_kept_and_discarded_groups_with_their_rewards(batch_policy):
    buffer = _buffer(batch_policy, batch_size=2, dynamic_sampling=DynamicSamplingType.FILTER)
    await buffer.publish(1)
    await _generate(buffer, "a")
    await _generate(
        buffer, "uniform", selection=GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, rewards=UNIFORM_REWARDS
    )
    await _generate(buffer, "masked", rejection=AdmissionRejection.FULLY_MASKED)
    await _generate(buffer, "a")
    await _generate(buffer, "b")

    while (admission := await buffer.admit(PROGRESS_TIMEOUT)).selection is None:
        pass
    assert admission.selection.judged == [
        JudgedGroup("a", (0.0, 1.0)),
        JudgedGroup("uniform", (0.0, 0.0)),
        JudgedGroup("b", (0.0, 1.0)),
    ]


@pytest.mark.asyncio
async def test_dynamic_sampling_fails_when_its_candidate_budget_is_exhausted(batch_policy):
    buffer = _buffer(batch_policy, batch_size=1, dynamic_sampling=DynamicSamplingType.FILTER, max_candidate_groups=2)
    await buffer.publish(1)
    uniform = {"selection": GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, "rewards": UNIFORM_REWARDS}
    await _generate(buffer, "first", **uniform)
    await _generate(buffer, "second", **uniform)

    with pytest.raises(RuntimeError, match="limit of 2 candidate groups"):
        await buffer.admit(PROGRESS_TIMEOUT)


@pytest.mark.asyncio
async def test_admission_without_progress_raises_a_stall_error(batch_policy):
    buffer = _buffer(batch_policy, batch_size=1)
    await buffer.publish(1)

    with pytest.raises(GroupAdmissionStalledError, match="admitted=0/1 leases=0"):
        await buffer.admit(BLOCKED_TIMEOUT)


@pytest.mark.parametrize(
    ("batch_policy", "first_batch"),
    [
        # "extra" was leased for batch 2 once the outstanding lease and "admitted" filled batch 1.
        (BatchPolicy.FULL_BATCH, ["admitted", "regenerated"]),
        (BatchPolicy.ROLLING, ["admitted", "extra"]),
    ],
)
@pytest.mark.asyncio
async def test_snapshot_restores_untaken_groups_and_reports_outstanding_leases(batch_policy, first_batch):
    buffer = _buffer(batch_policy, batch_size=2)
    await buffer.publish(1)
    outstanding = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await _generate(buffer, "admitted")
    await buffer.admit(PROGRESS_TIMEOUT)
    await _generate(buffer, "extra")

    snapshot = buffer.snapshot()
    assert snapshot.leases == {outstanding.lease_id}

    restored = _buffer(batch_policy, batch_size=2)
    await restored.restore(snapshot)
    await restored.publish(1)
    await _generate(restored, "regenerated")
    assert (await _take_batch(restored))[0] == first_batch


@pytest.mark.parametrize(
    ("batch_policy", "batches"),
    [
        (BatchPolicy.FULL_BATCH, [["a", "b"], ["c", "d"]]),
        (BatchPolicy.ROLLING, [["c", "d"], ["a", "b"]]),
    ],
)
@pytest.mark.asyncio
async def test_batch_policy_decides_whether_groups_train_in_lease_or_commit_order(batch_policy, batches):
    buffer = _buffer(batch_policy, batch_size=2, max_in_flight=4)
    await buffer.publish(1)
    leases = [await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT) for _ in range(4)]
    assert [lease.batch_id for lease in leases] == [1, 1, 2, 2]

    # The groups leased for batch 2 commit before the groups leased for batch 1.
    for index, uid in [(2, "c"), (3, "d"), (0, "a"), (1, "b")]:
        await _commit(buffer, leases[index].lease_id, uid)
    trained = [sorted((await _take_batch(buffer))[0])]
    await buffer.publish(2)
    trained.append(sorted((await _take_batch(buffer))[0]))
    assert trained == batches


@pytest.mark.asyncio
async def test_full_batch_waits_for_a_slow_group_instead_of_discarding_it():
    buffer = _buffer(BatchPolicy.FULL_BATCH, batch_size=1, max_in_flight=2)
    await buffer.publish(1)
    slow = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await _generate(buffer, "quick")
    with pytest.raises(GroupAdmissionStalledError):
        await buffer.admit(BLOCKED_TIMEOUT)

    await _commit(buffer, slow.lease_id, "slow")
    assert (await _take_batch(buffer))[0] == ["slow"]
    await buffer.publish(2)
    assert (await _take_batch(buffer))[0] == ["quick"]


@pytest.mark.asyncio
async def test_full_batch_regenerates_a_rejected_group_for_the_same_batch():
    buffer = _buffer(BatchPolicy.FULL_BATCH, batch_size=1, max_in_flight=4)
    await buffer.publish(1)
    masked = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert await _lease_is_blocked(buffer)

    await _commit(buffer, masked.lease_id, "masked", rejection=AdmissionRejection.FULLY_MASKED)
    replacement = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    assert replacement.batch_id == masked.batch_id == 1


@pytest.mark.asyncio
async def test_a_rejected_group_is_reported_before_the_batch_completes(batch_policy):
    buffer = _buffer(batch_policy, batch_size=2)
    await buffer.publish(1)
    await _generate(buffer, "masked", rejection=AdmissionRejection.FULLY_MASKED)

    admission = await buffer.admit(PROGRESS_TIMEOUT)
    assert [(outcome.uid, outcome.disposition) for outcome in admission.dispositions] == [("masked", "fully_masked")]
    assert admission.selection is None


@pytest.mark.asyncio
async def test_every_judged_group_reports_one_disposition_with_its_uid_and_dwell():
    buffer = _buffer(BatchPolicy.ROLLING, batch_size=1, dynamic_sampling=DynamicSamplingType.FILTER)
    await buffer.publish(1)
    stale = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
    dispositions = []
    for step, uid in enumerate(["a", "b"], start=2):
        await _generate(buffer, uid)
        while (admission := await buffer.admit(PROGRESS_TIMEOUT)).selection is None:
            dispositions.extend(admission.dispositions)
        dispositions.extend(admission.dispositions)
        await buffer.publish(step)

    await _commit(buffer, stale.lease_id, "stale")
    await _generate(buffer, "masked", rejection=AdmissionRejection.FULLY_MASKED)
    uniform = {"selection": GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD, "rewards": UNIFORM_REWARDS}
    await _generate(buffer, "uniform", **uniform)
    await _generate(buffer, "c")
    while (admission := await buffer.admit(PROGRESS_TIMEOUT)).selection is None:
        dispositions.extend(admission.dispositions)
    dispositions.extend(admission.dispositions)

    assert [(outcome.uid, outcome.disposition) for outcome in dispositions] == [
        ("a", "consumed"),
        ("b", "consumed"),
        ("stale", "stale"),
        ("masked", "fully_masked"),
        ("uniform", "insufficient_reward_spread"),
        ("c", "consumed"),
    ]
    assert all(outcome.tokens == 8 and outcome.dwell_seconds >= 0 for outcome in dispositions)


def _step_wise_group(uid, steps_per_trial, final_rewards):
    batch = {
        "response_ids": [],
        "prompt_token_ids": [],
        "loss_masks": [],
        "rewards": [],
        "unshaped_rewards": [],
        "is_last_step": [],
        "stop_reasons": [],
        "rollout_metrics": {},
    }
    for steps, reward in zip(steps_per_trial, final_rewards, strict=True):
        for step in range(steps):
            final = step == steps - 1
            batch["response_ids"].append([2])
            batch["prompt_token_ids"].append([1])
            batch["loss_masks"].append([1])
            batch["rewards"].append(reward if final else 0.0)
            batch["unshaped_rewards"].append(reward if final else 0.0)
            batch["is_last_step"].append(final)
            batch["stop_reasons"].append("stop")
    return RolloutGroup(batch, uid, 1, {})


@pytest.mark.asyncio
async def test_dynamic_sampling_judges_step_wise_groups_by_their_final_rows(batch_policy):
    content = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(DynamicSamplingType.FILTER),
    )
    buffer = _buffer(batch_policy, batch_size=2, dynamic_sampling=DynamicSamplingType.FILTER)
    await buffer.publish(1)
    for group in [
        _step_wise_group("three-rows", [2, 1], [1.0, 0.0]),
        _step_wise_group("five-rows", [3, 2], [0.0, 1.0]),
    ]:
        lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
        await buffer.commit(lease.lease_id, {}, content.verdict(group), PayloadReference(group.uid))
    while (admission := await buffer.admit(PROGRESS_TIMEOUT)).selection is None:
        pass
    assert admission.selection.judged == [JudgedGroup("three-rows", (1.0, 0.0)), JudgedGroup("five-rows", (0.0, 1.0))]
    assert admission.selection.metrics["async/dynamic_sampling/candidate_pass_at_2"] == 1.0
