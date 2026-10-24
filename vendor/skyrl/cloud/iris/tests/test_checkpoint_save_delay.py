"""Delayed saving applies to both periodic writes and early termination."""

import json
from types import SimpleNamespace

import pytest

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback, CheckpointCallback, HFModelSaveCallback


@pytest.mark.parametrize('total_steps, first_save', [(20, 10), (21, 11)])
@pytest.mark.parametrize('callback_type, flag', [
    (CheckpointCallback, 'should_save'), (HFModelSaveCallback, 'should_save_hf_model'),
])
def test_delayed_saving_respects_halfway_boundary_and_train_end(total_steps, first_save, callback_type, flag):
    callback = callback_type(save_steps=1, save_on_train_end=True, save_after_fraction=0.5)
    state = TrainerState(global_step=first_save - 1, epoch=0, total_steps=total_steps, num_steps_per_epoch=100)
    control = TrainerControl()
    callback.on_step_end(state, control)
    callback.on_train_end(state, control)
    assert not getattr(control, flag)
    state.global_step = first_save
    callback.on_step_end(state, control)
    assert getattr(control, flag)
    control = TrainerControl()
    callback.on_train_end(state, control)
    assert getattr(control, flag)


@pytest.mark.asyncio
async def test_best_checkpoint_ignores_improvements_before_save_boundary(tmp_path):
    state = TrainerState(global_step=0, epoch=0, total_steps=20, num_steps_per_epoch=20)

    class CheckpointWriter:
        cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path)))

        async def save_checkpoints(self):
            (tmp_path / f"global_step_{state.global_step}").mkdir()

    callback = BestCheckpointCallback(
        initial_model_uri="initial",
        expected_tasks=10,
        minimum_scored_tasks=10,
        save_after_fraction=0.5,
    )
    trainer = CheckpointWriter()
    for step, score in [(0, 0.1), (5, 0.9), (10, 0.5)]:
        state.global_step = step
        await callback.on_evaluate_async(
            state,
            TrainerControl(),
            trainer=trainer,
            metrics={
                "eval/all/num_attempted": 10,
                "eval/all/num_scored": 10,
                "eval/all/avg_verifier_score": score,
            },
        )
    assert not (tmp_path / "global_step_5").exists()
    assert (tmp_path / "global_step_10").is_dir()
    assert json.loads((tmp_path / "best_checkpoint.json").read_text())["step"] == 10
