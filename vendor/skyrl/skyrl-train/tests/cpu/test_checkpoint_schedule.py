"""Explicit checkpoint schedules and the masters-only optimizer filter."""

import json

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import CheckpointCallback, create_default_callbacks
from skyrl_train.checkpoint_schedule import (
    OPTIMIZER_STATE_MARKER,
    CheckpointSchedule,
    assert_optimizer_resumable,
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


def test_optimizer_steps_outside_checkpoint_steps_raise():
    cfg = schedule_config(
        checkpoint_steps=[10, 20, 30, 40, 50, 75, 100],
        optimizer_checkpoint_steps=[10, 60],
        ckpt_interval=0,
    )
    with pytest.raises(ValueError, match="steps that never save a checkpoint.*60"):
        CheckpointSchedule.from_config(cfg)


def test_optimizer_steps_outside_the_interval_raise():
    cfg = schedule_config(ckpt_interval=10, optimizer_checkpoint_steps=[12])
    with pytest.raises(ValueError, match="steps that never save a checkpoint.*12"):
        CheckpointSchedule.from_config(cfg)


def test_optimizer_steps_without_any_checkpoint_cadence_raise():
    cfg = schedule_config(ckpt_interval=0, optimizer_checkpoint_steps=[10])
    with pytest.raises(ValueError, match="no checkpoint ever saves"):
        CheckpointSchedule.from_config(cfg)


def test_max_gap_bounds_resume_consistency():
    listed = CheckpointSchedule.from_config(
        schedule_config(
            checkpoint_steps=[10, 20, 30, 40, 50, 75, 100],
            optimizer_checkpoint_steps=[20],
            ckpt_interval=0,
        )
    )
    assert listed.max_gap == 25

    intervalled = CheckpointSchedule.from_config(schedule_config(ckpt_interval=5))
    assert intervalled.max_gap == 5


def test_callback_saves_listed_steps_and_the_interval_fallback():
    listed = CheckpointCallback(CheckpointSchedule(interval=0, weight_steps=(10, 50), optimizer_steps=(10, 50)))
    assert listed.on_step_end(_state(10), _control()).should_save
    assert not listed.on_step_end(_state(20), _control()).should_save

    intervalled = CheckpointCallback(CheckpointSchedule(interval=10, weight_steps=(), optimizer_steps=()))
    assert intervalled.on_step_end(_state(20), _control()).should_save
    assert not intervalled.on_step_end(_state(13), _control()).should_save


def test_callback_saves_at_train_end():
    listed = CheckpointCallback(
        CheckpointSchedule(interval=0, weight_steps=(10,), optimizer_steps=(10,)), save_on_train_end=True
    )
    assert listed.on_train_end(_state(93), _control()).should_save


def _hf_save_callback(cfg):
    from skyrl_train.callbacks.builtin import HFModelSaveCallback

    matches = [callback for callback in create_default_callbacks(cfg) if isinstance(callback, HFModelSaveCallback)]
    assert len(matches) == 1
    return matches[0]


def test_hf_exports_fire_at_explicit_checkpoint_steps_not_the_interval():
    # The sweep's schedule: masters at 75 are not multiples of the rolling interval,
    # and interval exports between weight steps would crash on the missing marker.
    cfg = schedule_config(
        checkpoint_steps=[10, 20, 30, 40, 50, 75, 100],
        optimizer_checkpoint_steps=[10, 50, 100],
        ckpt_interval=10,
        export_hf_artifact=True,
    )
    callback = _hf_save_callback(cfg)
    fired = [step for step in range(1, 101) if callback.on_step_end(_state(step), _control()).should_save_hf_model]
    assert fired == [10, 20, 30, 40, 50, 75, 100]
    assert callback.on_train_end(_state(100), _control()).should_save_hf_model


def test_hf_exports_follow_the_scored_step_list_when_the_launcher_sets_one():
    # The launcher derives hf_export_steps from the checkpoints it scores; the
    # interval must not fire between them.
    cfg = schedule_config(
        checkpoint_steps=[10, 20, 30, 40, 50, 75, 100],
        optimizer_checkpoint_steps=[10, 50, 100],
        hf_export_steps=[10, 30, 50, 100],
        ckpt_interval=10,
        export_hf_artifact=True,
    )
    callback = _hf_save_callback(cfg)
    fired = [step for step in range(1, 101) if callback.on_step_end(_state(step), _control()).should_save_hf_model]
    assert fired == [10, 30, 50, 100]


def test_hf_export_steps_outside_the_checkpoint_schedule_are_rejected():
    cfg = schedule_config(
        checkpoint_steps=[10, 20, 30, 40, 50, 75, 100],
        optimizer_checkpoint_steps=[10, 50, 100],
        hf_export_steps=[10, 30, 60, 100],
        ckpt_interval=10,
        export_hf_artifact=True,
    )
    with pytest.raises(ValueError, match="hf_export_steps \\[60\\]"):
        _hf_save_callback(cfg)


def test_hf_exports_keep_the_interval_when_no_steps_are_listed():
    cfg = schedule_config(ckpt_interval=10, export_hf_artifact=True)
    callback = _hf_save_callback(cfg)
    fired = [step for step in range(1, 31) if callback.on_step_end(_state(step), _control()).should_save_hf_model]
    assert fired == [10, 20, 30]


def test_explicit_callbacks_conflict_with_checkpoint_steps():
    cfg = schedule_config(
        callbacks=[{"type": "checkpoint", "save_steps": 10}],
        checkpoint_steps=[10, 20, 30, 40, 50, 75, 100],
        optimizer_checkpoint_steps=[20],
    )
    with pytest.raises(ValueError, match="checkpoint_steps.*callbacks"):
        create_default_callbacks(cfg)


def test_explicit_callbacks_conflict_with_optimizer_checkpoint_steps():
    cfg = schedule_config(
        callbacks=[{"type": "checkpoint", "save_steps": 10}],
        optimizer_checkpoint_steps=[10],
    )
    with pytest.raises(ValueError, match="optimizer_checkpoint_steps.*callbacks"):
        create_default_callbacks(cfg)


def test_explicit_checkpoint_callback_builds_a_full_optimizer_schedule():
    cfg = schedule_config(callbacks=[{"type": "checkpoint", "save_steps": 10, "save_on_train_end": False}])
    callbacks = create_default_callbacks(cfg)
    checkpoint_callbacks = [callback for callback in callbacks if isinstance(callback, CheckpointCallback)]
    assert len(checkpoint_callbacks) == 1
    callback = checkpoint_callbacks[0]
    assert callback.on_step_end(_state(20), _control()).should_save
    assert callback.schedule.optimizer_mode(20) == "full"
    assert not callback.on_train_end(_state(93), _control()).should_save


def test_assert_optimizer_resumable_rejects_the_marker_on_any_path(tmp_path):
    step_dir = tmp_path / "global_step_20"
    marker = step_dir / "policy" / OPTIMIZER_STATE_MARKER
    marker.parent.mkdir(parents=True)
    assert_optimizer_resumable(str(step_dir))

    marker.write_text(json.dumps({"optimizer_state": "masters"}))
    with pytest.raises(ValueError, match="carries no optimizer state"):
        assert_optimizer_resumable(str(step_dir))

    marker.write_text(json.dumps({"optimizer_state": "full"}))
    assert_optimizer_resumable(str(step_dir))


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
