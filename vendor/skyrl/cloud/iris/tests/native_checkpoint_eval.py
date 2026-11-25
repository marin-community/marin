# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise checkpoint eligibility against a natively composed launch."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from cloud.iris import launch_config
from cloud.iris.config_cli import verify_checkout
from skyrl_train.callbacks.base import CallbackHandler, TrainerState, TrainerControl
from skyrl_train.callbacks.builtin import create_callbacks_from_config


def checkpoint_coverage(path: Path) -> list[dict | None]:
    config = launch_config.load_launch_config(path)
    assert not config.skyrl.generator.enable_http_endpoint
    state = TrainerState(global_step=0, epoch=0, total_steps=100, num_steps_per_epoch=100)
    scenarios = ((100, 99, 30, 27), (100, 98, 30, 30), (100, 100, 30, 26), (100, 100, 29, 29), (99, 99, 31, 30))
    results = []
    for index, (primary_attempted, primary_scored, bfcl_attempted, bfcl_scored) in enumerate(scenarios):
        directory = path.parent / f"coverage-{index}"
        config.skyrl.trainer.ckpt_path = str(directory)
        config.skyrl.trainer.export_path = str(directory / "exports")
        metrics = {
            "eval/all/num_attempted": primary_attempted,
            "eval/all/num_scored": primary_scored,
            "eval/all/avg_verifier_score": 0.25,
        }
        for name, attempted, scored, average in (
            ("bfcl_parity", bfcl_attempted, bfcl_scored, 1.0),
            ("swebench_verified", 30, 28, 0.5),
            ("openthoughts_tblite", 30, 30, 0.0),
        ):
            prefix = f"eval/{name}/all/"
            metrics.update(
                {
                    prefix + "num_attempted": attempted,
                    prefix + "num_scored": scored,
                    prefix + "avg_verifier_score": average,
                }
            )
        handler = CallbackHandler(create_callbacks_from_config(config.skyrl))
        asyncio.run(
            handler.call_event_async(
                "on_evaluate", state, TrainerControl(), metrics=metrics, trainer=SimpleNamespace(cfg=config.skyrl)
            )
        )
        record = directory / "best_checkpoint.json"
        results.append(json.loads(record.read_text()) if record.exists() else None)
    return results


if __name__ == "__main__":
    verify_checkout(Path(sys.argv[1]), launch_config)
    print(json.dumps(checkpoint_coverage(Path(sys.argv[2]))))
