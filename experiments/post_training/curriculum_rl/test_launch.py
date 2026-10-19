# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest
from marin.execution.lazy import StepContext
from marin.rl.skyrl import _launcher_command as launcher_command_for_test

from experiments.post_training.curriculum_rl import launch

COMPOSE_LAUNCHES = """
import json
import sys
from pathlib import Path
from cloud.iris.launch_config import load_launch_config

configs = [load_launch_config(Path(path)) for path in sys.argv[1:]]
print(json.dumps([
    {"profile": config.runtime.profile, "strategy": config.skyrl.trainer.strategy}
    for config in configs
]))
"""


@pytest.mark.timeout(180)  # A cold cache installs the independently locked CPU launcher.
@pytest.mark.parametrize(
    ("policy", "preset"),
    [(launch.QWEN_POLICY, launch.SMOKE), (launch.QWEN_POLICY, launch.FULL)]
    + [(launch.SNOWBALL_POLICY, preset) for preset in launch.SCALES.values() if preset.label.startswith("snowball-")],
    ids=lambda value: value.label,
)
def test_curriculum_arms_compose_in_nested_runtime(policy, preset, tmp_path) -> None:
    # Marin previously accepted FSDP recipes that the real launcher rejected before submission.
    # Validate every sampler arm at the executable boundary without allocating GPUs.
    arms = launch.build_arms(
        specs=tuple(launch.ARMS.values()), scale=preset.label, policy=policy, version="2026.10.19.1"
    )
    paths = []
    for name, arm in arms.items():
        config = arm.rl.build_config(StepContext.for_fingerprint(arm.rl.runtime_args, arm.rl.deps))
        path = tmp_path / f"{name}.yaml"
        path.write_text(config.launch_config_yaml)
        paths.append(str(path))
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run(
        [*bootstrap, "python", "-c", COMPOSE_LAUNCHES, *paths], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [{"profile": "megatron", "strategy": "megatron"}] * len(arms)
