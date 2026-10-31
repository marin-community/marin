# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish bounded CPU receipts locally and to the experiment artifact prefix."""

import json
import os
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.mini_policy_runtime.profile import DESTINATION


def publish(name, data, destination=DESTINATION):
    content = json.dumps(data, indent=2).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / name).write_bytes(content)
    fs, key = filesystem_for(str(StoragePath(destination) / name))
    fs.pipe_file(key, content)
