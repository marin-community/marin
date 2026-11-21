import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from loguru import logger
from omegaconf import OmegaConf
from rolloutengine.models import ModelTurn, RolloutFailure
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.config.utils import get_default_config

from skyrl_train.rollouts.buffer import BatchPolicy, RolloutBuffer, RolloutBufferConfig
from skyrl_train.rollouts.task_worker import HarborSlots, admission_slot
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingStep, TrainingTransition, reject_capped_rollout
from skyrl_train.telemetry import process_telemetry
from skyrl_train.trainer import RayPPOTrainer


@pytest.mark.asyncio
async def test_harbor_slots_admit_oldest_waiting_batch_before_earlier_arrivals(telemetry_endpoint):
    slots = HarborSlots(1)
    entered = []

    async def run(batch_id, name):
        async with slots.admit(batch_id):
            entered.append(name)

    with process_telemetry("worker"):
        async with slots.admit(0):
            waiters = []
            for batch_id, name in [(4, "four"), (3, "three"), (1, "one-a"), (1, "one-b")]:
                waiters.append(asyncio.create_task(run(batch_id, name)))
                await asyncio.sleep(0)  # Each arrival must queue before the next one is submitted.
            cancelled = asyncio.create_task(run(0, "cancelled"))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert entered == []
        await asyncio.gather(*waiters)
    assert entered == ["one-a", "one-b", "three", "four"]
    resumed = telemetry_endpoint.select("rollout_admission_resumed")
    assert [event["body"]["batch_id"] for event in resumed] == [1, 1, 3, 4]


@pytest.mark.asyncio
async def test_harbor_slot_grant_cancelled_before_resume_returns_capacity():
    slots = HarborSlots(1)
    entered = []

    async def run():
        async with slots.admit(1):
            entered.append(1)

    async with slots.admit(0):
        pending = asyncio.create_task(run())
        await asyncio.sleep(0)
    pending.cancel()  # The free slot has been granted, but its coroutine has not resumed.
    with pytest.raises(asyncio.CancelledError):
        await pending
    async with slots.admit(2):
        entered.append(2)
    assert entered == [2]


@pytest.mark.parametrize(
    "stop_reason,failure,diagnostics",
    [
        ("length", None, {}),
        ("total_turn_timeout", None, {}),
        ("error", RolloutFailure("AgentTimeoutError"), {}),
        ("stop", None, {"harbor_exception_type": "AgentTimeoutError"}),
        ("stop", None, {"harbor_model_limit": "length"}),
    ],
)
def test_capped_trajectory_rejects_positive_verifier_and_turn_rewards(stop_reason, failure, diagnostics):
    turn = ModelTurn(
        prompt_token_ids=(1,), response_token_ids=(2,), logprobs=(-0.1,),
        message={"role": "assistant", "content": "answer"}, stop_reason="stop",
    )
    grade = GradeResult(Outcome.GRADED, 1.0, passed=True, diagnostics=diagnostics)
    rollout = TrainingRollout(
        task_id="capped", messages=(), prompt_token_ids=(1,), response_token_ids=(2,),
        loss_mask=(1,), logprobs=(-0.1,), grade=grade, stop_reason=stop_reason, failure=failure,
        steps=(TrainingStep(turn, TrainingTransition(
            done=True, reward=1.0, token_rewards=(1.0,), token_credit=(1.0,), grade=grade,
        ), 0, ()),),
    )
    rejected = reject_capped_rollout(rollout)
    assert (rejected.grade.status, rejected.grade.reward, rejected.grade.passed) == (Outcome.SUBMISSION_FAILURE, 0.0, False)
    assert rejected.steps[0].transition.reward is None
    assert rejected.steps[0].transition.token_rewards is None
    assert rejected.steps[0].transition.token_credit is None
    completed = replace(rollout, stop_reason="stop", failure=None, grade=GradeResult(Outcome.GRADED, 1.0, passed=True))
    assert reject_capped_rollout(completed).grade.reward == 1.0


@pytest.mark.asyncio
async def test_blocked_trajectory_slot_exports_wait_and_preserves_capacity_on_cancel(telemetry_endpoint):
    slots = asyncio.Semaphore(1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def occupy():
        async with admission_slot("harbor_trials", slots):
            entered.set()
            await release.wait()

    with process_telemetry("worker"):
        await slots.acquire()
        blocked = asyncio.create_task(occupy())
        await asyncio.sleep(0)
        assert not entered.is_set()
        slots.release()
        await entered.wait()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        assert not slots.locked()
        async with admission_slot("harbor_trials", slots):
            pass
    [waiting] = telemetry_endpoint.select("rollout_admission_blocked")
    [resumed] = telemetry_endpoint.select("rollout_admission_resumed")
    assert waiting["body"]["gate"] == resumed["body"]["gate"] == "harbor_trials"
    assert resumed["body"]["wait_seconds"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("max_in_flight,staleness,gate", [(1, 1, "max_in_flight"), (8, 0, "batch_lookahead")])
async def test_generation_limit_records_which_gate_prevents_another_lease(max_in_flight, staleness, gate):
    records = []
    sink = logger.add(lambda message: records.append(message.record["extra"]))
    buffer = RolloutBuffer(RolloutBufferConfig(1, max_in_flight, staleness, BatchPolicy.FULL_BATCH, None, None))
    try:
        await buffer.publish(1)
        await buffer.acquire_lease()
        waiting = asyncio.create_task(buffer.acquire_lease())
        await asyncio.sleep(0)
        assert not waiting.done()
        assert records[-1]["admission_gate"] == gate
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        logger.remove(sink)


def test_startup_audit_exports_effective_budget_and_incidental_limits(telemetry_endpoint):
    cfg = get_default_config()
    cfg.generator.n_samples_per_prompt = 8
    cfg.trajectory_runner.rollout_workers.num_workers = 32
    cfg.environment.task_sessions.max_verifier_workers = 64
    OmegaConf.update(cfg, "terminal_bench_config.harbor.n_concurrent_trials", 384, force_add=True)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer._training_metrics_enabled = True
    trainer.context = SimpleNamespace(
        config=RolloutBufferConfig(32, 1024, 3, BatchPolicy.FULL_BATCH, None, None,
                                   {"agentic": 16, "nonagentic": 16})
    )
    trainer.global_step = 0
    trainer.distillation_plan = None
    with process_telemetry("trainer"):
        trainer._record_run_configuration()
    [event] = telemetry_endpoint.select("rollout_concurrency_configuration")
    body = event["body"]
    assert body["max_in_flight_groups"] == 1024
    assert body["effective_in_flight_groups"] == body["lookahead_groups"] == 128
    assert body["trajectory_bound_per_worker"] == 32
    assert body["verifier_threads_per_worker"] == 64
    assert body["harbor_trials_per_worker"] == 12
    assert body["symbolic_math_slots_per_process"] == 8
    assert body["teacher_enabled"] is False
