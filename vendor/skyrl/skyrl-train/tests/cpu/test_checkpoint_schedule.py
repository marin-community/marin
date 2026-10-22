"""Explicit checkpoint schedules and the masters-only optimizer filter."""

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import CheckpointCallback
from skyrl_train.checkpoint_schedule import (
    CheckpointSchedule,
    master_only_optimizer_state,
)
from skyrl_train.config.utils import get_default_config


def schedule_config(**overrides) -> OmegaConf:
    cfg = get_default_config()
    return OmegaConf.merge(cfg, OmegaConf.create({"trainer": overrides}))


def test_explicit_steps_replace_the_interval_and_restrict_the_optimizer():
    cfg = schedule_config(
        checkpoint_steps=[20, 10, 20],
        optimizer_checkpoint_steps=[10],
        ckpt_interval=0,
    )
    schedule = CheckpointSchedule.from_config(cfg)
    assert schedule.should_save_weights(10) and schedule.should_save_weights(20)
    assert not schedule.should_save_weights(15) and not schedule.should_save_weights(30)
    assert schedule.optimizer_mode(10) == "full"
    assert schedule.optimizer_mode(20) == "masters"
    assert not schedule.should_save_optimizer(20)


def test_empty_lists_follow_the_interval_and_keep_every_optimizer():
    cfg = schedule_config(ckpt_interval=5)
    schedule = CheckpointSchedule.from_config(cfg)
    assert schedule.should_save_weights(10) and not schedule.should_save_weights(7)
    assert schedule.should_save_optimizer(7)
    assert schedule.optimizer_mode(7) == "full"


def test_weight_steps_require_an_explicit_optimizer_list():
    cfg = schedule_config(checkpoint_steps=[10, 20, 30, 40, 50, 75, 100])
    with pytest.raises(ValueError, match="optimizer_checkpoint_steps"):
        CheckpointSchedule.from_config(cfg)


def test_step_lists_reject_non_positive_entries():
    cfg = schedule_config(checkpoint_steps=[10, 0], optimizer_checkpoint_steps=[10])
    with pytest.raises(ValueError, match="positive step numbers"):
        CheckpointSchedule.from_config(cfg)


def test_callback_saves_listed_steps_and_the_interval_fallback():
    listed = CheckpointCallback(save_step_list=[10, 50])
    assert listed.on_step_end(_state(10), _control()).should_save
    assert not listed.on_step_end(_state(20), _control()).should_save

    interval = CheckpointCallback(save_steps=10)
    assert interval.on_step_end(_state(20), _control()).should_save
    assert not interval.on_step_end(_state(13), _control()).should_save


def test_callback_saves_at_train_end():
    listed = CheckpointCallback(save_step_list=[10])
    control = listed.on_train_end(_state(93), _control())
    assert control.should_save


def test_master_only_optimizer_state_keeps_params_and_masters():
    optimizer_state = {
        "optimizer": {
            "state": {
                "fp32_param.module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight": "master",
                "param.module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight": "param",
                "exp_avg.module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight": "moment",
                "exp_avg_sq.module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight": "moment",
            },
            "step": 100,
        },
        "rng": "state",
    }
    filtered = master_only_optimizer_state(optimizer_state)
    kept = filtered["optimizer"]["state"]
    assert set(kept) == {
        "fp32_param.module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight",
        "param.module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight",
    }
    assert "step" not in filtered["optimizer"]
    assert "rng" not in filtered


def test_master_only_optimizer_state_requires_master_records():
    with pytest.raises(ValueError, match="no param or FP32 master records"):
        master_only_optimizer_state({"optimizer": {"state": {"exp_avg.module.x": "moment"}, "step": 1}})


def _state(step: int) -> TrainerState:
    return TrainerState(
        global_step=step,
        epoch=0,
        total_steps=100,
        num_steps_per_epoch=100,
        is_last_step=False,
        is_epoch_end=False,
        metrics={},
        timings={},
    )


def _control() -> TrainerControl:
    return TrainerControl()
