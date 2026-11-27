# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from marinskyrl.runtime_options import BatchBuilder
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import CallbackHandler, TrainerCallback, TrainerControl
from skyrl_train.callbacks.builtin import EvaluationCallback
from skyrl_train.timing_observability import STEP_WALL_PHASES, StepWallTime
from skyrl_train.trainer import RayPPOTrainer


class DispatchProbe:
    def __init__(self):
        self.config = SimpleNamespace(batch_size=1, max_staleness_steps=1)
        self.loop = asyncio.get_running_loop()
        self.paused = False

    async def next_batch_metadata(self, *, stall_timeout):
        assert not self.paused
        return SimpleNamespace(metrics={}, groups=())

    async def next_batch(self, *, stall_timeout, on_admitted):
        return [], await self.next_batch_metadata(stall_timeout=stall_timeout)

    def reader(self, metadata, *, stall_timeout):
        return None

    def pause(self):
        assert asyncio.get_running_loop() is self.loop
        self.paused = True

    def resume(self):
        assert asyncio.get_running_loop() is self.loop
        self.paused = False


class CancelEvaluation(TrainerCallback):
    def on_step_end(self, state, control, **kwargs):
        return TrainerControl(should_log=False)


class BatchAssemblyReached(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_builder", [BatchBuilder.DRIVER, BatchBuilder.WORKER])
@pytest.mark.parametrize(
    ("step", "has_eval_data", "pauses"),
    [(2, True, True), (1, True, False), (2, False, False)],
    ids=["evaluating", "regular", "no-eval-data"],
)
async def test_evaluation_step_pauses_dispatch_after_the_training_batch(
    batch_builder, step, has_eval_data, pauses, monkeypatch
):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.batch_builder = batch_builder
    trainer.context = DispatchProbe()
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
    trainer.cfg = OmegaConf.create({"trainer": {"policy": {"megatron_config": {}}}})
    trainer._step_time_history = []
    trainer.group_admission_stall_timeout = 1
    trainer._rollout_spans_enabled = False
    trainer._speculator_capture_active = False
    trainer._sealed_speculator_capture_uri = None
    trainer.colocate_all = False
    trainer._distillation_runtime = None
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer.global_step = step
    trainer.total_training_steps = trainer.num_steps_per_epoch = 4
    trainer.eval_dataset = object() if has_eval_data else None
    trainer._control = TrainerControl()
    trainer.callback_handler = CallbackHandler([EvaluationCallback(eval_steps=2), CancelEvaluation()])

    def assemble_batch(*args, **kwargs):
        raise BatchAssemblyReached

    trainer.convert_rollout_groups_to_training_input = assemble_batch
    monkeypatch.setattr("skyrl_train.trainer.plan_worker_batch", assemble_batch)
    with pytest.raises(BatchAssemblyReached):
        await trainer._train_step(0, StepWallTime(dict.fromkeys(STEP_WALL_PHASES, 0.0)))
    assert trainer.context.paused is pauses
    await trainer._run_step_end_callbacks(trainer._create_trainer_state(0))
    assert not trainer.context.paused
