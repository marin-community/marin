import json
from types import SimpleNamespace

import pytest

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback


@pytest.mark.asyncio
async def test_delayed_best_selection_preserves_baseline_gates_after_resume(tmp_path):
    saved_steps = []
    current_step = 0

    async def save_checkpoints():
        saved_steps.append(current_step)
        (tmp_path / f'weights-{current_step}').write_text('checkpoint')

    trainer = SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path))),
                              save_checkpoints=save_checkpoints)
    parameters = dict(initial_model_uri='base-model', expected_tasks=40, minimum_scored_tasks=40,
                      selection_metric='identity', save_after_fraction=0.5,
                      preservation_max_drop={'preservation': 0.0})
    callback = BestCheckpointCallback(**parameters)

    async def evaluate(step, identity, preservation):
        nonlocal current_step
        current_step = step
        return await callback.on_evaluate_async(
            TrainerState(step, 0, 20, 20), TrainerControl(), trainer=trainer,
            metrics={'eval/all/num_attempted': 40, 'eval/all/num_scored': 40,
                     'identity': identity, 'preservation': preservation})

    await evaluate(0, 0.1, 0.9)
    await evaluate(5, 1.0, 0.9)
    assert saved_steps == []
    await evaluate(10, 0.5, 0.9)
    assert saved_steps == [10]
    callback = BestCheckpointCallback(**parameters)
    await evaluate(15, 0.8, 0.85)
    assert saved_steps == [10]
    await evaluate(20, 0.7, 0.9)
    assert saved_steps == [10, 20]
    await evaluate(20, 0.7, 0.9)
    assert saved_steps == [10, 20]
    selected = json.loads((tmp_path / 'best_checkpoint.json').read_text())
    assert selected['step'] == 20
    assert selected['accuracy'] == 0.7
    assert selected['preservation_baseline'] == {'preservation': 0.9}
    assert selected['model_format'] == 'native'
