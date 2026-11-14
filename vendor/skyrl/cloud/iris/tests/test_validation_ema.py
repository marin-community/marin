import json
from types import SimpleNamespace

import pytest
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback

from marinskyrl.checkpoint_paths import trailing_validation_ema


def test_trailing_five_rewards_excludes_old_peak_and_requires_complete_window():
    rows = [{"step": i, "reward": r} for i, r in enumerate([1, 0, 0, 0, 0, 1], 1)]
    weights = [(2 / 3) ** n for n in [4, 3, 2, 1, 0]]
    assert trailing_validation_ema(rows, 6, 5) == pytest.approx(1 / sum(weights))
    assert trailing_validation_ema(rows[:4], 4, 5) is None
    assert trailing_validation_ema([r for r in rows if r["step"] != 3], 6, 5) is None


@pytest.mark.asyncio
async def test_best_callback_persists_window_and_restarts_without_changing_tie(tmp_path):

    saved = []

    class Trainer:
        cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path)))

        async def save_checkpoints(self):
            saved.append(state.global_step)

    trainer = Trainer()
    callback = BestCheckpointCallback(initial_model_uri="base", expected_tasks=2, minimum_scored_tasks=2, ema_window=5)
    control = TrainerControl()
    for step in range(1, 7):
        if step == 6:
            callback = BestCheckpointCallback(initial_model_uri="base", expected_tasks=2, minimum_scored_tasks=2, ema_window=5)
        state = TrainerState(global_step=step, epoch=0, total_steps=6, num_steps_per_epoch=6)
        await callback.on_evaluate_async(state, control, trainer=trainer, metrics={
            "eval/all/num_attempted": 2, "eval/all/num_scored": 2, "eval/all/avg_verifier_score": 0.5,
        })
    assert saved == [5]
    assert json.loads((tmp_path / "best_checkpoint.json").read_text())["step"] == 5
    assert len(json.loads((tmp_path / "validation_selection_history.json").read_text())) == 6

    with pytest.raises(ValueError, match="already recorded"):
        await callback.on_evaluate_async(state, control, trainer=trainer, metrics={
            "eval/all/num_attempted": 2, "eval/all/num_scored": 2, "eval/all/avg_verifier_score": 0.7,
        })


def test_five_observation_ema_at_two_step_cadence():
    rows = [{"step": step, "reward": 0.5} for step in (2, 4, 6, 8, 10)]
    assert trailing_validation_ema(rows, 10, 5, interval=2) == pytest.approx(0.5)
    assert trailing_validation_ema(rows[1:], 10, 5, interval=2) is None
