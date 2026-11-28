# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch document's training schedule the way the native trainer would."""

import json
import sys
from pathlib import Path

from cloud.iris import launch_config
from cloud.iris.config_cli import verify_checkout
from skyrl_train.trainer import RayPPOTrainer

verify_checkout(Path(sys.argv[1]), launch_config)
path = Path(sys.argv[2])
config = launch_config.load_launch_config(path)
trainer = RayPPOTrainer.__new__(RayPPOTrainer)
trainer.cfg = config.skyrl
trainer.train_dataset = range(100000)
trainer._configure_training_schedule()
print(json.dumps({"stop": trainer.total_training_steps, "available": trainer.available_training_steps}))
