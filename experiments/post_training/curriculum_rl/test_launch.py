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


@pytest.mark.timeout(180)
def test_snowball_curriculum_smoke_composes_separate_reference_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    step = launch.build_arms(
        specs=(launch.ARMS["naive"],), scale="snowball-smoke", policy=launch.SNOWBALL_POLICY, version="2026.10.15"
    )["naive"].rl
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    path = tmp_path / "reference.yaml"
    path.write_text(config.launch_config_yaml)
    command = launcher_command_for_test(str(path))
    result = subprocess.run(
        [
            *command[: command.index("marinskyrl")],
            "python",
            "-c",
            """
import json, sys
from cloud.iris.launch_config import load_launch_config
config = load_launch_config(sys.argv[1])
placement = config.skyrl.trainer.placement
assert not placement.colocate_policy_ref
print(json.dumps({
    "policy_gpus": placement.policy_num_nodes * placement.policy_num_gpus_per_node,
    "reference_gpus": placement.ref_num_nodes * placement.ref_num_gpus_per_node,
    "allocated_gpus": config.iris.allocation.num_nodes * config.iris.allocation.gpus_per_node,
}))
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=170,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"policy_gpus": 32, "reference_gpus": 16, "allocated_gpus": 56}
