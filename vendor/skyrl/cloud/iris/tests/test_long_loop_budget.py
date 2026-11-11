"""Explicit budgets start at optimizer completion, without a minimum-step failure."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import WallTimeBudgetCallback


def test_budget_excludes_startup_and_stops_with_durable_evidence(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('skyrl_train.callbacks.builtin.time.monotonic', lambda: clock[0])
    trainer = SimpleNamespace(cfg=OmegaConf.create({'trainer': {'export_path': str(tmp_path)}}),
                              first_optimizer_step_time=1000.0)
    callback = WallTimeBudgetCallback(seconds=100, drain_margin=20, groups_per_step=64)
    control = TrainerControl()
    callback.on_train_begin(TrainerState(0, 0, 100, 100), control)
    clock[0] = 1105.0
    callback.on_log(TrainerState(1, 0, 100, 100), control, logs={}, trainer=trainer)
    assert control.should_training_stop
    report = json.loads((tmp_path / 'training-budget.json').read_text())
    assert report['elapsed_seconds'] == 105.0
    assert report['consumed_groups_per_hour'] == pytest.approx(64 * 3600 / 105)
    assert report['additional_updates_at_observed_rate'] == 0


def test_budget_does_not_count_pre_optimizer_startup(tmp_path, monkeypatch):
    monkeypatch.setattr('skyrl_train.callbacks.builtin.time.monotonic', lambda: 100000.0)
    trainer = SimpleNamespace(cfg=OmegaConf.create({'trainer': {'export_path': str(tmp_path)}}),
                              first_optimizer_step_time=None)
    callback = WallTimeBudgetCallback(seconds=100, drain_margin=20, groups_per_step=64)
    control = TrainerControl()
    callback.on_train_begin(TrainerState(0, 0, 100, 100), control)
    callback.on_log(TrainerState(0, 0, 100, 100), control, logs={}, trainer=trainer)
    assert not control.should_training_stop
    assert not (tmp_path / 'training-budget.json').exists()
