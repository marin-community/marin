"""Budget stops preserve completed updates and reserve export time."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import WallTimeBudgetCallback


def test_measured_rate_stops_before_next_update_and_keeps_durable_evidence(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('skyrl_train.callbacks.builtin.time.monotonic', lambda: clock[0])
    trainer = SimpleNamespace(cfg=OmegaConf.create({'trainer': {'export_path': str(tmp_path)}}))
    callback = WallTimeBudgetCallback(seconds=100, reserve=20, minimum_steps=5, groups_per_step=64)
    control = TrainerControl()
    callback.on_train_begin(TrainerState(0, 0, 100, 100), control)
    clock[0] = 70.0
    callback.on_log(TrainerState(5, 0, 100, 100), control, logs={}, trainer=trainer)
    assert control.should_training_stop
    report = json.loads((tmp_path / 'training-budget.json').read_text())
    assert report['consumed_groups_per_hour'] == pytest.approx(320 * 3600 / 70)
    assert report['additional_updates_at_observed_rate'] == 0


def test_budget_failure_does_not_export_an_incomplete_ema_window(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('skyrl_train.callbacks.builtin.time.monotonic', lambda: clock[0])
    trainer = SimpleNamespace(cfg=OmegaConf.create({'trainer': {'export_path': str(tmp_path)}}))
    callback = WallTimeBudgetCallback(seconds=100, reserve=20, minimum_steps=5, groups_per_step=64)
    control = TrainerControl()
    callback.on_train_begin(TrainerState(0, 0, 100, 100), control)
    clock[0] = 70.0
    with pytest.raises(RuntimeError, match='minimum validation-selection window'):
        callback.on_log(TrainerState(2, 0, 100, 100), control, logs={}, trainer=trainer)
