from rigging import timing
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import CheckpointCallback


def test_timed_checkpoint_resets_after_save_and_forces_finalization(monkeypatch):
    now = 0.0
    monkeypatch.setattr(timing.time, "monotonic", lambda: now)
    callback = CheckpointCallback(save_steps=-1, save_interval_seconds=7200, save_on_train_end=True)
    state = TrainerState(global_step=1, epoch=0, total_steps=100, num_steps_per_epoch=100)
    control = TrainerControl()
    callback.on_train_begin(state, control)
    now = 7199
    assert not callback.on_step_end(state, control).should_save
    now = 7200
    assert callback.on_step_end(state, control).should_save
    callback.on_save(state, control)
    control.reset()
    now = 14399
    assert not callback.on_step_end(state, control).should_save
    assert callback.on_train_end(state, control).should_save
