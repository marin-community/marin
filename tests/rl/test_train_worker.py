# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
