import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger
from ray.exceptions import GetTimeoutError

from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV
import skyrl_train.trainer as trainer_module
from skyrl_train.trainer import RayPPOTrainer


class _UnpickleableError(RuntimeError):
    def __reduce__(self):
        raise pickle.PicklingError("exception cannot be pickled")


def test_model_initialization_phase_waits_only_for_remaining_budget(monkeypatch):
    waits = []

    def get(refs, timeout):
        waits.append(timeout)
        raise GetTimeoutError("workers still downloading")

    monkeypatch.setattr(trainer_module.ray, "get", get)
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(monotonic=lambda: 100.0), raising=False)

    with pytest.raises(RuntimeError, match="timed out"):
        trainer_module._wait_for_setup_phase(
            ["policy-worker-ref"],
            deadline=1100.0,
            phase="policy/ref/critic model initialization",
        )

    # The phase waits only for the remaining budget.
    assert waits == [1000.0]


@pytest.mark.asyncio
async def test_startup_failure_still_runs_trainer_shutdown():
    events = []
    trainer = object.__new__(RayPPOTrainer)
    trainer._shutdown_complete = False
    trainer.global_step = 0
    trainer._distillation_runtime = None
    trainer.context = SimpleNamespace(close=AsyncMock())

    async def fail_startup():
        events.append("startup")
        raise RuntimeError("runner failed to start")

    async def teardown():
        events.append("teardown")

    trainer._startup_trajectory_runner = fail_startup
    trainer._teardown = teardown

    with pytest.raises(RuntimeError, match="runner failed to start"):
        await trainer.train()

    assert events == ["startup", "teardown"]


@pytest.mark.asyncio
async def test_trainer_shutdown_is_idempotent():
    events = []
    trainer = object.__new__(RayPPOTrainer)
    trainer._shutdown_complete = False
    trainer.context = SimpleNamespace(close=AsyncMock())

    async def teardown():
        events.append("teardown")

    trainer._teardown = teardown

    await trainer.shutdown()
    await trainer.shutdown()

    assert events == ["teardown"]


@pytest.mark.asyncio
async def test_training_failure_preserves_receipt_before_shutdown(monkeypatch, tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_step = 12
    trainer._distillation_runtime = None
    trainer.context = SimpleNamespace(close=AsyncMock())
    trainer.trajectory_runner = SimpleNamespace(startup=AsyncMock())
    trainer.all_startup_timings = {}
    trainer.eval_trajectory_runner = trainer.trajectory_runner
    trainer._train_loop = AsyncMock(side_effect=_UnpickleableError("GPU worker ran out of memory"))
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path))

    async def teardown():
        receipts = list((tmp_path / "outcomes").glob("skyrl-trainer.*.exception.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["exception_type"] == f"{_UnpickleableError.__module__}.{_UnpickleableError.__qualname__}"
        assert receipt["message"] == "GPU worker ran out of memory"

    trainer._teardown = teardown
    messages = []
    sink_id = logger.add(messages.append, level="ERROR")

    try:
        with pytest.raises(_UnpickleableError, match="GPU worker ran out of memory"):
            await trainer.train()
    finally:
        logger.remove(sink_id)

    assert messages
    assert all(message.record["exception"] is None for message in messages)
    for message in messages:
        pickle.dumps(message.record)
