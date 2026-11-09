# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate concrete graph configs before allocating remote workers."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from marin.rl.skyrl import skyrl_cpu_command


def validate_launch(document: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "launch.yaml"
    source.write_text(document)
    compiled = directory / "compiled.yaml"
    command = skyrl_cpu_command(
        "python", "-m", "experiments.post_training.long_loop.validate_launch", str(source), str(compiled)
    )
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    (directory / "validator.log").write_text(result.stderr)
    if result.returncode:
        raise ValueError(f"Strict SkyRL composition failed before allocation:\n{result.stderr}")
    receipt = json.loads(result.stdout)
    receipt["command"] = command
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def validate_before_allocation(document: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="long-loop-validation-") as directory:
        return validate_launch(document, Path(directory))
