import asyncio
from types import SimpleNamespace

import pytest
import ray

from skyrl_train import rollout_observability
from skyrl_train.config.utils import get_default_config
from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts import context as context_module
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutLease,
    RolloutVerdict,
)
from skyrl_train.task_inputs import PreferenceInput, TrajectoryMetadata
from skyrl_train.telemetry import GeneratedWork
from tests.cpu.task_specs import lowered_task, task_input, text_task


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


class DispatchDataset:
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index]

    def uid(self, index: int) -> str:
        return self.inputs[index].metadata.uid

    label = uid

    def collate_fn(self, items: list) -> list[dict]:
        return [{"uid": item.metadata.uid, "input": item} for item in items]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("preference_pairs", "expected"),
    [(False, [("lease", 1), ("task", 1)]), (True, [("slot", 1), ("prompt", 1)])],
    ids=["tasks", "preference-pairs"],
)
async def test_dispatch_counts_one_wait_per_phase(monkeypatch, preference_pairs, expected):
    config = get_default_config()
    config.trainer.train_batch_size = 1
    config.trainer.rollout_buffer.max_outstanding_leases = 1
    config.trainer.rollout_spans = True
    if preference_pairs:
        config.environment.env_class = "preference_pair"
        inputs = [PreferenceInput([1], {}, TrajectoryMetadata(uid, "fixture")) for uid in ("a", "b")]
    else:
        inputs = [task_input(lowered_task(text_task(uid), "text")) for uid in ("a", "b")]

    def local_buffer(buffer_config):
        buffer = RolloutBuffer(buffer_config)
        return SimpleNamespace(
            acquire_lease=SimpleNamespace(remote=buffer.acquire_lease),
            publish=SimpleNamespace(remote=buffer.publish),
        )

    observed = []

    def count(value, *, attributes):
        observed.append((attributes["wait"], value))

    dispatched = asyncio.Event()

    async def run_group_job(job, writer):
        dispatched.set()
        return 0

    monkeypatch.setattr(context_module, "start_rollout_buffer", local_buffer)
    monkeypatch.setattr(ray, "kill", lambda _: None)
    monkeypatch.setattr(rollout_observability, "waits", SimpleNamespace(add=count))
    training = context_module.TrainingContext.from_config(
        config, DispatchDataset(inputs), SimpleNamespace(run_group_job=run_group_job)
    )
    training.start()
    try:
        async with asyncio.timeout(5):
            await training.publish(1)
            await training.wait(dispatched.wait())
        assert observed == expected
    finally:
        await training.close()
