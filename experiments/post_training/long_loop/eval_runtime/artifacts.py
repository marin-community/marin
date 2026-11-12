# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish bounded CPU receipts locally and to the experiment artifact prefix."""

import json
import os
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.eval_runtime.profile import DESTINATION


def publish(name, data, destination=DESTINATION):
    content = json.dumps(data, indent=2).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / name).write_bytes(content)
    (StoragePath(destination) / name).write_bytes(content)
