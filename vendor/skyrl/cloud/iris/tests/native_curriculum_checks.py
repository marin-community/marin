# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the curriculum's training budget and checkpoint window inside SkyRL."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from cloud.iris.config_cli import verify_checkout
from omegaconf import OmegaConf
from skyrl_train.callbacks import builtin
from skyrl_train.callbacks.base import TrainerControl, TrainerState


def schedule(checkout: Path, path: Path) -> dict:
    from skyrl_train import trainer as training

    verify_checkout(checkout, training)
    trainer = training.RayPPOTrainer.__new__(training.RayPPOTrainer)
    trainer.cfg = OmegaConf.create(json.loads(path.read_text())["skyrl"])
    trainer.train_dataset = range(100000)
    trainer._configure_training_schedule()
    return {"stop": trainer.total_training_steps, "available": trainer.available_training_steps}


async def checkpoint_window(root: Path) -> dict:
    saved = []
    step = 0

    async def save_checkpoints():
        saved.append(step)

    trainer = SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(root), export_path=str(root / "exports"))),
        save_checkpoints=save_checkpoints,
    )
    params = dict(initial_model_uri="sft-final", expected_tasks=200, minimum_scored_tasks=198, selection_window=3)
    callback = builtin.BestCheckpointCallback(**params)
    windows = []
    for step, score in zip((3, 6, 9, 12, 15, 18), (0.9, 0.1, 0.2, 0.8, 0.8, 0.1), strict=True):
        if step == 15:
            callback = builtin.BestCheckpointCallback(**params)
        metrics = {"eval/all/num_attempted": 200, "eval/all/num_scored": 200, "eval/all/avg_verifier_score": score}
        await callback.on_evaluate_async(TrainerState(step, 0, 18, 18), TrainerControl(), trainer=trainer, metrics=metrics)
        windows.append(metrics.get("eval/selection_score"))
    return {
        "selected": json.loads((root / "best_checkpoint.json").read_text()),
        "saved": saved,
        "windows": windows,
        "export": json.loads((root / "exports/best_checkpoint.json").read_text()),
    }


if __name__ == "__main__":
    checkout, operation, path = sys.argv[1:]
    verify_checkout(Path(checkout), builtin)
    result = schedule(Path(checkout), Path(path)) if operation == "schedule" else asyncio.run(checkpoint_window(Path(path)))
    print(json.dumps(result))
