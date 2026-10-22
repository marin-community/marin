"""Bound periodic probe capture across rollout groups and policy publications."""

import asyncio
import inspect
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import numpy as np
import ray
from ray.actor import ActorHandle
from omegaconf import OmegaConf
from rolloutengine.models import ModelTurn
from taskcompendium.grading_result import GradeResult, Outcome
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutLease,
    RolloutTask,
    RolloutWriter,
)
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext
from skyrl_train.rollouts.loader import PromptLoader, PromptOrder, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, PayloadStore
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingStep, TrainingTransition
from skyrl_train.trajectory_runners.types import TaskRequestBatch

from tests.cpu.task_specs import lowered_task, task_input, text_task

SAMPLES_PER_PROMPT = 2


STALL_TIMEOUT = 10.0


CONTENT_POLICY = RolloutContentPolicy(
    GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=SAMPLES_PER_PROMPT),
        rollout_logprobs_required=False,
    ),
    GroupSelectionPolicy(None),
)


class _Prompts:
    def __init__(self, uids: list[str]):
        self._uids = uids

    def __len__(self) -> int:
        return len(self._uids)

    def __getitem__(self, index: int) -> str:
        return self._uids[index]

    def uid(self, index: int) -> str:
        return self._uids[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [_prompt(uid) for uid in items]


def _prompt(uid: str) -> dict:
    return {"uid": uid, "input": task_input(lowered_task(text_task(uid)))}


def _batch() -> dict:
    return {
        "prompt_token_ids": [[1]] * SAMPLES_PER_PROMPT,
        "response_ids": [[2]] * SAMPLES_PER_PROMPT,
        "rewards": [0.0, 1.0],
        "loss_masks": [[1]] * SAMPLES_PER_PROMPT,
        "rollout_logprobs": None,
    }


class _Workers:
    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        await writer.write_rollout(
            task.lease, RolloutGroup(_batch(), task.prompt["uid"], task.lease.policy_step, task.prompt)
        )
        return SAMPLES_PER_PROMPT


class _InProcessActor:
    """Serves an actor's ``method.remote(...)`` calls on the caller's event loop.

    Starting a real buffer actor spawns a Ray worker that imports the trainer, which takes seconds per test.
    """

    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name: str) -> SimpleNamespace:
        method = getattr(self._instance, name)

        async def call(*args):
            result = method(*args)
            return await result if inspect.isawaitable(result) else result

        return SimpleNamespace(remote=lambda *args: asyncio.ensure_future(call(*args)))


def _in_process_buffer(config: RolloutBufferConfig) -> _InProcessActor:
    return _InProcessActor(RolloutBuffer(config))


StartBuffer = Callable[[RolloutBufferConfig], ActorHandle]


@pytest.fixture
def start_buffer(monkeypatch) -> StartBuffer:
    # An in-process buffer has no actor for ``TrainingContext.close`` to kill.
    monkeypatch.setattr(ray, "kill", lambda _actor: None)
    return _in_process_buffer


def _context(
    uids: list[str],
    workers: _Workers,
    start_buffer: StartBuffer,
    *,
    batch_size: int,
    max_in_flight: int,
    max_staleness_steps: int = 1,
    batch_policy: BatchPolicy = BatchPolicy.FULL_BATCH,
    order: PromptOrder | None = None,
    payloads: PayloadStore | None = None,
    request_spec: RolloutRequestSpec | None = None,
) -> TrainingContext:
    config = RolloutBufferConfig(batch_size, max_in_flight, max_staleness_steps, batch_policy, None, None)
    return TrainingContext(
        PromptLoader(_Prompts(uids), order or SeededPasses(len(uids), seed=0, shuffle=False), batch_size=batch_size),
        config,
        start_buffer(config),
        CONTENT_POLICY,
        request_spec or RolloutRequestSpec(samples_per_prompt=SAMPLES_PER_PROMPT, sampling_params={}),
        workers,
        payloads or MemoryPayloads(),
        rollout_spans=False,
    )


async def _ignore(groups: list[RolloutGroup]) -> None:
    pass


async def _next_uids(context: TrainingContext) -> list[str]:
    groups, _ = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
    return [group.uid for group in groups]


@pytest.mark.asyncio
@pytest.mark.parametrize("topk", [0, 128])
async def test_training_task_persists_capture_costs_with_its_rollout(topk):
    ids = np.tile(np.arange(128, dtype=np.int32), (4, 1))
    scores = np.full((4, 128), -np.log(128), dtype=np.float32)
    logprobs = tuple(float(value) for value in scores[:, 0])
    response = (11, 12, 13, 14)
    messages = ({"role": "assistant", "content": "answer"},)
    turn = ModelTurn(
        message=messages[0], text="answer", prompt_token_ids=(1,), response_token_ids=response,
        logprobs=logprobs, stop_reason="stop",
        metadata={"student_topk_indices": ids, "behavior_topk_logprobs": scores} if topk else {},
    )
    grade = GradeResult(status=Outcome.GRADED, reward=1.0)
    outputs = [TrainingRollout(
        task_id="prompt", messages=messages, prompt_token_ids=(1,), response_token_ids=response,
        loss_mask=(1, 1, 1, 1), logprobs=logprobs, grade=grade, stop_reason="stop",
        steps=(TrainingStep(turn, TrainingTransition(done=True, reward=1.0, grade=grade), 3, messages),),
    )]

    class Worker(TaskRolloutWorker):
        async def generate(self, request):
            return outputs

    class Writer:
        async def write_rollout(self, lease, group):
            self.group = group

    worker = object.__new__(Worker)
    worker.closing, worker.active_requests = False, set()
    worker.trajectory_runner_cfg = OmegaConf.create({"apply_overlong_filtering": False, "sampling_params": {"logprobs": topk}})
    worker.projection = WholeTaskProjection(worker.trajectory_runner_cfg, SimpleNamespace(eos_token_id=2))
    worker.trajectory_sink = None
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(text_task("prompt"))),), sampling_params={"logprobs": topk},
        score_centering_probe_tokens=1 if topk else 0,
    )
    task = RolloutTask(RolloutLease("lease", 7, 8), {"uid": "prompt"}, request)
    writer = Writer()
    assert await worker.run_task(task, writer) == 4
    assert writer.group.policy_step == 7
    metrics = writer.group.trajectory_batch.get("rollout_metrics", {})
    if topk:
        assert metrics["score_centering_capture/requests"] == 1
        assert metrics["score_centering_capture/probe_requests"] == 1
        assert metrics["score_centering_capture/captured_tokens"] == 4
        assert metrics["score_centering_capture/captured_head_bytes"] == 4096
        assert metrics["score_centering_capture/exported_head_bytes"] == 1028
        assert metrics["score_centering_capture/collection_seconds_sum"] >= 0
        assert metrics["score_centering_capture/assembly_seconds_sum"] >= 0
    else:
        assert not any(name.startswith("score_centering_capture/") for name in metrics)


@pytest.mark.asyncio
@pytest.mark.parametrize("interval,expected_steps", [(0, []), (2, [1, 3, 5]), (3, [1, 4])])
async def test_periodic_probe_dispatch_is_bounded_across_many_groups_and_repeated_steps(
    ray_module, start_buffer, interval, expected_steps
):
    requests = []

    class Workers(_Workers):
        async def run_task(self, task, writer):
            requests.append(task.request)
            return await super().run_task(task, writer)

    context = _context(
        [str(index) for index in range(16)],
        Workers(),
        start_buffer,
        batch_size=3,
        max_in_flight=3,
        max_staleness_steps=0,
        request_spec=RolloutRequestSpec(
            SAMPLES_PER_PROMPT, {"logprobs": 0}, probe_interval=interval, probe_topk=128, probe_tokens=32
        ),
    )
    context.start()
    try:
        for step in range(1, 6):
            await context.publish(step)
            assert len(await _next_uids(context)) == 3
        sampled = [request for request in requests if request.score_centering_probe_tokens]
        assert [request.batch_metadata.global_step for request in sampled] == expected_steps
        assert all(
            request.score_centering_probe_tokens == 32 and request.sampling_params["logprobs"] == 128
            for request in sampled
        )
        assert all(
            request.sampling_params["logprobs"] == 0
            for request in requests
            if not request.score_centering_probe_tokens
        )
    finally:
        await context.close()
