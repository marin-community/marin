# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run bounded runtime qualification and retain its receipt even on failure."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from marin.rl.skyrl import skyrl_cpu_command
from rigging.filesystem.storage_path import StoragePath, prefix_join


def run_qualification(
    *,
    output: str,
    inputs: dict | list,
    module: str,
    prefix_arguments: tuple[str, ...] = (),
    dependency_group: str | None = None,
    extra: str = "cpu",
    environment_overrides: dict[str, str] | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="long-loop-qualification-") as temporary:
        root = Path(temporary)
        source, receipt = root / "inputs.json", root / "receipt.json"
        source.write_text(json.dumps(inputs))
        command = skyrl_cpu_command(
            *(("--group", dependency_group) if dependency_group else ()),
            "python",
            *prefix_arguments,
            "-m",
            module,
            "--inputs",
            str(source),
            "--output",
            str(receipt),
        )
        command[command.index("--extra") + 1] = extra
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
            environment.pop(key, None)
        environment.update(environment_overrides or {})
        try:
            with (root / "probe.log").open("w") as log:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        finally:
            for path in (source, root / "probe.log", receipt):
                if path.exists():
                    StoragePath(prefix_join(output, path.name)).write_bytes(path.read_bytes())
        result.check_returncode()
        if not json.loads(receipt.read_text())["passed"]:
            raise ValueError(f"Runtime qualification failed: {module}")
