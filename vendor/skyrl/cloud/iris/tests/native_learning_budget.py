# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive the authored wall-time budget callback across its startup, running and expired phases."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cloud.iris.config_cli import verify_checkout
from omegaconf import OmegaConf
from skyrl_train.callbacks import builtin
from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState

verify_checkout(Path(sys.argv[1]), builtin)
path = Path(sys.argv[2])
config = OmegaConf.load(path)
config.trainer.export_path = str(path.parent)
receipt = path.parent / builtin.TRAINING_BUDGET_RECEIPT
handler = CallbackHandler([builtin.create_callback_from_config(row) for row in config.trainer.callbacks])
trainer = SimpleNamespace(cfg=config, first_optimizer_step_time=None)
control = TrainerControl()


async def event(name, step, **kwargs):
    await handler.call_event_async(
        name, TrainerState(global_step=step, epoch=0, total_steps=100, num_steps_per_epoch=100), control,
        trainer=trainer, **kwargs
    )


asyncio.run(event("on_train_begin", 7))
with patch("skyrl_train.callbacks.builtin.time.monotonic", return_value=100000):
    asyncio.run(event("on_log", 7, logs={}))
startup_stop = control.should_training_stop
assert not receipt.exists()
trainer.first_optimizer_step_time = 1000.0
with patch("skyrl_train.callbacks.builtin.time.monotonic", return_value=1099):
    asyncio.run(event("on_log", 8, logs={}))
before_stop = control.should_training_stop
with patch("skyrl_train.callbacks.builtin.time.monotonic", return_value=1105):
    asyncio.run(event("on_log", 9, logs={}))
print(
    json.dumps(
        {
            "startup_stop": startup_stop,
            "before_stop": before_stop,
            "expired_stop": control.should_training_stop,
            "receipt": json.loads(receipt.read_text()),
        }
    )
)
