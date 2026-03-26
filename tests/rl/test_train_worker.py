# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from marin.rl.rl_losses import RLOOLoss
from marin.rl.train_worker import TrainWorker


def test_drop_bootstrap_model_references_clears_reference_model_when_kl_disabled():
    worker = TrainWorker.__new__(TrainWorker)
    model = object()
    worker.loss_module = RLOOLoss(kl_coef=0.0)
    worker.initial_model = model
    worker.reference_model = model

    worker._drop_bootstrap_model_references()

    assert worker.initial_model is None
    assert worker.reference_model is None


def test_drop_bootstrap_model_references_preserves_reference_model_when_kl_enabled():
    worker = TrainWorker.__new__(TrainWorker)
    model = object()
    worker.loss_module = RLOOLoss(kl_coef=0.01)
    worker.initial_model = model
    worker.reference_model = model

    worker._drop_bootstrap_model_references()

    assert worker.initial_model is None
    assert worker.reference_model is model


def test_record_train_step_updates_replay_buffer_and_shared_run_state():
    recorded_steps: list[int] = []

    class _FakeRemoteMethod:
        def remote(self, step: int) -> None:
            recorded_steps.append(step)

    class _FakeRunState:
        update_train_step = _FakeRemoteMethod()

    worker = TrainWorker.__new__(TrainWorker)
    worker.replay_buffer = SimpleNamespace(set_current_step=recorded_steps.append)
    worker._runtime = SimpleNamespace(run_state=_FakeRunState())

    worker._record_train_step(7)

    assert recorded_steps == [7, 7]
