# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from marin.rl.skyrl import skyrl_cpu_command
from rigging.config_discovery import find_project_root

from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as snowball


def current_control_pins(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "CONTROLS",
        {name: replace(pin, revision=snowball.PRESETS[name].revision) for name, pin in gate.CONTROLS.items()},
    )


def selection_summary(full_rows: int) -> dict:
    return {
        "recipe": {"heldout": [{"version": "2026.10.23.1", "rows": 910}]},
        "training_rows": 100000,
        "validation_rows": 100,
        "full_validation_rows": full_rows,
        "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
    }


def native_json():
    def execute(script, paths):
        source = str(find_project_root(Path(__file__)) / "vendor/skyrl")
        program = (
            "from pathlib import Path as SourcePath\nfrom cloud.iris import launch_config as SourceLaunch\n"
            "import skyrl_train as SourceTrain\n"
            f"assert all(SourcePath(module.__file__).resolve().is_relative_to({source!r}) "
            "for module in (SourceLaunch, SourceTrain))\n"
            "print('SNOWBALL_NATIVE_SOURCE_ORIGINS_PASS')\n" + script
        )
        result = subprocess.run(
            skyrl_cpu_command("python", "-c", program, *paths), capture_output=True, text=True, timeout=170
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    return execute
