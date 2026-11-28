# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a best-checkpoint selection window: the mean of the last completed evaluations."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from cloud.iris.config_cli import verify_checkout
from skyrl_train.callbacks import builtin
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback

verify_checkout(Path(sys.argv[1]), builtin)
root = Path(sys.argv[2])
params = dict(initial_model_uri="sft-final", expected_tasks=200, minimum_scored_tasks=198, selection_window=3)


async def main():
    saved = []
    current = {"step": 0}

    async def save_checkpoints():
        saved.append(current["step"])

    trainer = SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(root), export_path=str(root / "exports"))),
        save_checkpoints=save_checkpoints,
    )
    callback = BestCheckpointCallback(**params)
    windows = []
    for step, score in zip((3, 6, 9, 12, 15, 18), (0.9, 0.1, 0.2, 0.8, 0.8, 0.1)):
        current["step"] = step
        if step == 15:
            # A restart rebuilds the callback; the window restores from the persisted history.
            callback = BestCheckpointCallback(**params)
        metrics = {
            "eval/all/num_attempted": 200,
            "eval/all/num_scored": 200,
            "eval/all/avg_verifier_score": score,
        }
        state = TrainerState(global_step=step, epoch=0, total_steps=18, num_steps_per_epoch=18)
        await callback.on_evaluate_async(state, TrainerControl(), trainer=trainer, metrics=metrics)
        windows.append(metrics.get("eval/selection_score"))
    selected = json.loads((root / "best_checkpoint.json").read_text())
    print(
        json.dumps(
            {
                "selected": selected,
                "saved": saved,
                "windows": windows,
                "export": json.loads((root / "exports/best_checkpoint.json").read_text()),
            }
        )
    )


asyncio.run(main())
