# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.rl.skyrl import _launcher_command as launcher_command_for_test

from experiments.post_training.phase1a.masters_gate import gate_step, reload_recipe, save_recipe
from experiments.post_training.phase1a.test_phase1a import COMPOSE


@pytest.mark.timeout(300)  # A cold cache installs the independently locked launcher.
def test_the_gate_saves_masters_only_then_full_and_reloads_at_zero_learning_rate(tmp_path):
    paths = []
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.24"))):
        for name, recipe in (("save", save_recipe()), ("reload", reload_recipe("s3://bucket/ckpt/global_step_3"))):
            step = gate_step(name, recipe, "s3://bucket/b", "2026.10.22", "s3://bucket/rows", "2026.10.24")
            config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
            path = tmp_path / f"{name}.yaml"
            path.write_text(config.launch_config_yaml)
            paths.append(str(path))
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run([*bootstrap, "python", "-c", COMPOSE, *paths], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    save, reload = json.loads(result.stdout)
    assert save["checkpoints"] == [[2, 3], [3]] and save["steps"][0] == 3
    assert reload["checkpoints"] == [[4], [4]] and reload["optimizer"][1] == 0.0
    assert save["optimizer"][5] < save["steps"][0]  # Megatron asserts warmup < decay steps
