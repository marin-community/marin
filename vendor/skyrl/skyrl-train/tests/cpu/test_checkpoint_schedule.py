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
    master_only_optimizer_state,
)
from skyrl_train.config.utils import get_default_config

MODEL_KEY = "module.decoder.layers.0.mlp.experts.expert0.linear_fc1.weight"


class ShardedRecord:
    """A DCP record stub named the way MCore names optimizer ShardedTensors."""

    def __init__(self, key: str):
        self.key = key


def schedule_config(**overrides) -> OmegaConf:
    cfg = get_default_config()
    return OmegaConf.merge(cfg, OmegaConf.create({"trainer": overrides}))


def fully_reshardable_optimizer_state() -> dict:
    """Optimizer state shaped like MCore's fully_reshardable output for one parameter.

    The dict paths (``optimizer.param_state.0.fp32_param``) do not name DCP records;
    only ``leaf.key`` does.
    """
    return {
        "optimizer": {"param_groups": ShardedRecord("optimizer.param_groups")},
        "param_state": {
            0: {
                "fp32_param": ShardedRecord(f"optimizer.state.fp32_param.{MODEL_KEY}"),
                "param": ShardedRecord(f"chained_0.optimizer.state.param.{MODEL_KEY}"),
                "exp_avg": ShardedRecord(f"optimizer.state.exp_avg.{MODEL_KEY}"),
                "exp_avg_sq": ShardedRecord(f"optimizer.state.exp_avg_sq.{MODEL_KEY}"),
            },
        },
        "param_state_sharding_type": "fully_reshardable",
    }


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


def test_optimizer_steps_require_the_fully_reshardable_format():
    cfg = schedule_config(
        ckpt_interval=10,
        optimizer_checkpoint_steps=[10],
    )
    cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = "dp_reshardable"
    with pytest.raises(ValueError, match="fully_reshardable"):
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


def test_master_only_optimizer_state_filters_on_leaf_keys():
    """Dict paths name dict positions; only ShardedTensor.key names the DCP records."""
    filtered = master_only_optimizer_state(fully_reshardable_optimizer_state())
    kept = filtered["param_state"][0]
    assert set(kept) == {"fp32_param", "param"}
    assert kept["fp32_param"].key == f"optimizer.state.fp32_param.{MODEL_KEY}"
    assert kept["param"].key == f"chained_0.optimizer.state.param.{MODEL_KEY}"
    assert "optimizer" not in filtered
    assert "param_state_sharding_type" not in filtered


def test_master_only_optimizer_state_requires_master_records():
    dp_reshardable = {
        "param_state": {
            0: {
                "param": ShardedRecord(
                    "optimizer.distributed.dp_group_idx_0.gbuf_idx_0."
                    "dtype_torch.bfloat16.bucket_idx_0.param"
                ),
                "exp_avg": ShardedRecord(
                    "optimizer.distributed.dp_group_idx_0.gbuf_idx_0."
                    "dtype_torch.bfloat16.bucket_idx_0.exp_avg"
                ),
            },
        },
        "param_state_sharding_type": "dp_reshardable",
    }
    with pytest.raises(ValueError, match="no param or FP32 master records"):
        master_only_optimizer_state(dp_reshardable)


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
