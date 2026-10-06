# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Russell task workers with the isolated rollout dependencies."""

import json
import subprocess
import tempfile
from pathlib import Path

from rigging.config_discovery import find_project_root


def run_task_worker(module: str, values: dict, *, arguments: tuple[str, ...] = ()) -> None:
    """Run a task worker with its config in a temporary JSON file."""
    workspace = find_project_root()
    if workspace is None:
        raise RuntimeError("Task construction requires the bundled Marin workspace")
    with tempfile.TemporaryDirectory(prefix="russell-task-config-") as directory:
        config = Path(directory) / "config.json"
        config.write_text(json.dumps(values))
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(workspace / "lib/rolloutengine"),
                "--with",
                "openai==2.24.0",
                "--with-editable",
                str(workspace / "lib/iris"),
                "python",
                "-m",
                module,
                *arguments,
                "--config",
                str(config),
            ],
            cwd=workspace,
            check=True,
        )
