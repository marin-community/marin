"""Aborted groups cannot commit, and evaluation waits for sandbox teardown."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from rolloutengine.cleanup import finish_cleanup
from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.config.rollout_validation import FullBatchFill
from skyrl_train.rollouts.buffer import (
    BatchPolicy, BatchFillDeadlineExceeded, GroupRewards, PayloadReference, RolloutBuffer, RolloutBufferConfig, RolloutVerdict,
)
from skyrl_train.rollouts.context import EvalDrainStrategy, RolloutRequestSpec, TrainingContext
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads
from skyrl_train.rollouts.workers import RolloutWorker
from skyrl_train.telemetry import GeneratedWork, process_telemetry
from tests.cpu.task_specs import lowered_task, task_input, text_task


class Dataset:
    def __len__(self):
        return 2

    def uid(self, index):
        return str(index)

    def __getitem__(self, index):
        return {"uid": str(index), "input": task_input(lowered_task(text_task(), "agent"))}

    def collate_fn(self, rows):
        return rows


def verdict(uid):
    return RolloutVerdict(uid, (), GroupSelectionResult.KEEP, GroupRewards((1.,), (1.,), True), GeneratedWork(1, 1, 1))


def local_actor(runner):
    # Exercise the actor methods without replacing their task tracking/cleanup
    # barrier. Only the accelerator/provider runner and RPC transport are local.
    actor_class = RolloutWorker.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor._runner = runner
    actor._training_tasks = {}
    actor._aborted_leases = set()
    return actor


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [EvalDrainStrategy.WAIT, EvalDrainStrategy.ABORT, EvalDrainStrategy.PRIORITY])
@pytest.mark.parametrize("bounded_fill", [False, True])
async def test_eval_drain_preserves_or_discards_groups_and_acknowledges_sandbox_cleanup(
    monkeypatch, telemetry_endpoint, strategy, bounded_fill
):
    buffer = RolloutBuffer(RolloutBufferConfig(
        1, 2, 1, BatchPolicy.FULL_BATCH, None, None,
        full_batch_fill=FullBatchFill(600, 900) if bounded_fill else None,
    ))
    started = asyncio.Event()
    finish_generation = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    deleted = []
    scored = []
    attempts = Counter()
    requests = []
    remote_tasks = []

    async def provider_run(task, writer):
        uid = task.prompt["uid"]
        requests.append(task)
        attempts[uid] += 1
        if uid == "1" and attempts[uid] == 1:
            started.set()

            async def delete_sandbox():
                cleanup_started.set()
                await release_cleanup.wait()
                deleted.append(uid)

            try:
                await finish_generation.wait()
            finally:
                # The real engine/Harbor cleanup also survives repeated cancels.
                await finish_cleanup(delete_sandbox, timeout=5)
        scored.append(uid)
        await buffer.commit(task.lease.lease_id, task.prompt, verdict(uid), PayloadReference(uid))
        return 1

    actor = local_actor(SimpleNamespace(run_task=provider_run))

    async def remote_run(task, writer):
        remote = asyncio.create_task(actor.run_task(task, writer, False))
        remote_tasks.append(remote)
        try:
            result, _ = await asyncio.shield(remote)
            return result
        except asyncio.CancelledError:
            remote.cancel()
            raise  # Like a Ray await: driver cancellation precedes remote cleanup.

    async def snapshot():
        return buffer.snapshot()

    actor_buffer = SimpleNamespace(**{
        name: SimpleNamespace(remote=method) for name, method in {
            "acquire_lease": buffer.acquire_lease, "snapshot": snapshot, "publish": buffer.publish,
            "admit": buffer.admit,
            "abort_leases": buffer.abort_leases, "resume_leasing": buffer.resume_leasing,
            "bind_prompt": buffer.bind_prompt,
        }.items()
    })
    context = TrainingContext(
        PromptLoader(Dataset(), SeededPasses(2, seed=0, shuffle=False), batch_size=1, max_stale_retries=0, max_unusable_groups=4,
                     sources={str(i): Dataset()[i]["input"].metadata.data_source for i in range(2)}),
        buffer.config, actor_buffer, None, RolloutRequestSpec(1, {}),
        SimpleNamespace(run_task=remote_run, abort_training=actor.abort_training), MemoryPayloads(),
        rollout_spans=False, eval_drain=strategy,
    )
    monkeypatch.setattr("skyrl_train.rollouts.context.ray.kill", lambda actor: None)
    # Step-zero evaluation runs before the training dispatcher exists.
    await context.pause_and_drain()
    assert requests == []
    await context.resume()
    context.start()
    try:
        await context.publish(1)
        await asyncio.wait_for(started.wait(), timeout=5)
        batch = await context.next_batch_metadata(stall_timeout=1)
        assert [group.uid for group in batch.groups] == ["0"]
        aborted_request = next(task for task in requests if task.prompt["uid"] == "1")
        if strategy is EvalDrainStrategy.PRIORITY:
            await context.prepare_evaluation()
            await context.pause_and_drain()
            assert context._submission_allowed.is_set()
            assert not cleanup_started.is_set()
            assert scored == ["0"]
            assert buffer.snapshot().ready == []
            finish_generation.set()
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
            release_cleanup.set()
            await context.publish(2)
            next_batch = await context.next_batch_metadata(stall_timeout=1)
            assert [group.uid for group in next_batch.groups] == ["1"]
            assert attempts["1"] == 1
            return
        if strategy is EvalDrainStrategy.WAIT:
            drain = asyncio.create_task(context.pause_and_drain())
            await asyncio.sleep(0)  # Let drain reach its wait without advancing a wall clock.
            assert not drain.done()
            assert not cleanup_started.is_set()
            finish_generation.set()
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
            assert not drain.done()
            release_cleanup.set()
            await asyncio.wait_for(drain, timeout=5)
            assert scored == ["0", "1"]
            assert [group.prompt["uid"] for group in buffer.snapshot().ready] == ["1"]
            await context.resume()
            await context.publish(2)
            next_batch = await context.next_batch_metadata(stall_timeout=1)
            assert [group.uid for group in next_batch.groups] == ["1"]
            assert attempts["1"] == 1
            return
        with process_telemetry("trainer"):
            abort = asyncio.create_task(context.prepare_evaluation())
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
            assert not abort.done()
            # A result that arrives after revocation is discarded even if it
            # already carries a valid verdict and nonempty payload.
            await buffer.commit(aborted_request.lease.lease_id, aborted_request.prompt, verdict("1"), PayloadReference("late"))
            assert buffer.snapshot().ready == []
            state = await context.state_dict()
            assert [row["uid"] for row in state.loader.retries] == ["1"]
            release_cleanup.set()
            await asyncio.wait_for(abort, timeout=5)
        assert deleted == ["1"]
        assert scored == ["0"]
        [event] = telemetry_endpoint.select("rollout_evaluation_aborted")
        assert event["body"]["aborted_groups"] == 1
        assert event["body"]["active_groups"] == 1
        # An RPC queued behind the abort must not recreate its sandbox.
        with pytest.raises(asyncio.CancelledError):
            await actor.run_task(aborted_request, None, False)
        assert attempts["1"] == 1
        await context.resume()
        await context.publish(2)
        replacement = await context.next_batch_metadata(stall_timeout=1)
        assert [group.uid for group in replacement.groups] == ["1"]
        assert attempts["1"] == 2
        assert next(task for task in requests if task.prompt["uid"] == "1" and task is not aborted_request).lease != aborted_request.lease
    finally:
        release_cleanup.set()
        await context.close()
        for task in remote_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*remote_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_timeout", [False, True])
async def test_absolute_batch_deadline_covers_admission_callback_without_hiding_io_timeout(monkeypatch, payload_timeout):
    config = RolloutBufferConfig(1, 1, 0, BatchPolicy.FULL_BATCH, None, None,
                               full_batch_fill=FullBatchFill(.01, .1))
    buffer = RolloutBuffer(config)
    transport = SimpleNamespace(**{name: SimpleNamespace(remote=method) for name, method in {
        "publish": buffer.publish, "admit": buffer.admit,
    }.items()})
    context = TrainingContext(
        PromptLoader(Dataset(), SeededPasses(2, seed=0, shuffle=False), batch_size=1,
                     max_stale_retries=0, max_unusable_groups=4),
        config, transport, None, RolloutRequestSpec(1, {}), SimpleNamespace(), MemoryPayloads(),
        rollout_spans=False, eval_drain=EvalDrainStrategy.PRIORITY,
    )
    monkeypatch.setattr("skyrl_train.rollouts.context.ray.kill", lambda actor: None)
    context.pause()
    context.start()
    entered = asyncio.Event()
    hold_payload = asyncio.Event()

    async def read_payload(groups):
        assert [group.uid for group in groups] == ["0"]
        entered.set()
        if payload_timeout:
            raise TimeoutError("payload transport timeout")
        await hold_payload.wait()

    try:
        await context.publish(1)
        lease = await buffer.acquire_lease()
        row = Dataset()[0]
        await buffer.bind_prompt(lease.lease_id, row)
        await buffer.commit(lease.lease_id, row, verdict("0"), PayloadReference("0"))
        expected = TimeoutError if payload_timeout else BatchFillDeadlineExceeded
        with pytest.raises(expected) as error:
            await context.next_batch_metadata(stall_timeout=10000, on_admitted=read_payload)
        assert entered.is_set()
        if payload_timeout:
            assert type(error.value) is TimeoutError
    finally:
        await context.close()
