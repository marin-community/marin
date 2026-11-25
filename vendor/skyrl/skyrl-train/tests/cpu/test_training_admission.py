# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from marinskyrl.runtime_options import BatchBuilder
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import CallbackHandler, TrainerControl
from skyrl_train.callbacks.builtin import EvaluationCallback
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.trainer import RayPPOTrainer


class AdmissionProbe(TrainingContext):
    def __init__(self):
        self.config = SimpleNamespace(batch_size=1)
        self._submission_allowed = asyncio.Event()
        self._submission_allowed.set()

    async def next_batch_metadata(self, *, stall_timeout):
        assert self._submission_allowed.is_set()
        return SimpleNamespace(metrics={})

    async def next_batch(self, *, stall_timeout, on_admitted):
        return [], await self.next_batch_metadata(stall_timeout=stall_timeout)

    def reader(self, metadata, *, stall_timeout):
        return None


class BatchAssemblyReached(Exception):
    pass


class BatchAssemblyBoundary:
    def start(self, phase):
        if phase == "batch_assembly":
            raise BatchAssemblyReached


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_builder", [BatchBuilder.DRIVER, BatchBuilder.WORKER])
async def test_evaluation_step_pauses_admission_after_the_training_batch(batch_builder):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.batch_builder = batch_builder
    trainer.context = AdmissionProbe()
    trainer.policy_model = None
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
    trainer.global_step = 2
    trainer.total_training_steps = trainer.num_steps_per_epoch = 4
    trainer.eval_dataset = object()
    trainer._control = TrainerControl()
    trainer.callback_handler = CallbackHandler([EvaluationCallback(eval_steps=2)])

    with pytest.raises(BatchAssemblyReached):
        await trainer._train_step(0, BatchAssemblyBoundary())
    assert not trainer.context._submission_allowed.is_set()
