# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
from marin.execution.lazy import StepContext
from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training.inventory.train import build_run

SOURCE_URIS = {"training_uri": "s3://test/training.parquet", "validation_uri": "s3://test/validation.parquet"}


@pytest.mark.timeout(180)
@pytest.mark.parametrize("steps", [3, 25, 50])
def test_short_and_normal_runs_compose_native_training_and_terminal_outputs(steps: int, tmp_path: Path) -> None:
    # Exercise the runtime boundary: a three-update run must retain initial/final
    # native-task evaluation and an aligned terminal checkpoint/export schedule.
    run = build_run("2026.10.22.2", steps=steps, **SOURCE_URIS)
    config = run.build_config(StepContext.for_fingerprint(run.runtime_args, run.deps))
    path = tmp_path / "launch.yaml"
    path.write_text(config.launch_config_yaml)
    script = """
import json
import sys
from cloud.iris.launch_config import load_launch_config

launch = load_launch_config(sys.argv[1])
trainer = launch.skyrl.trainer
print(json.dumps({
    "entrypoint": launch.runtime.entrypoint,
    "training_type": launch.runtime.training_type,
    "initial_evaluation": trainer.eval_before_train,
    "updates": trainer.max_steps,
    "evaluation_interval": trainer.eval_interval,
    "checkpoint_interval": trainer.ckpt_interval,
    "hf_interval": trainer.hf_save_interval,
    "export_hf": launch.run.export_hf,
    "all_heldout_prompts": trainer.eval_num_prompts is None,
}))
"""
    result = subprocess.run(
        skyrl_cpu_command("python", "-c", script, str(path)),
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"}
        },
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "entrypoint": "skyrl_train.entrypoints.taskcompendium",
        "training_type": "sync",
        "initial_evaluation": True,
        "updates": steps,
        "evaluation_interval": steps,
        "checkpoint_interval": steps,
        "hf_interval": steps,
        "export_hf": True,
        "all_heldout_prompts": True,
    }


def test_changing_training_or_data_does_not_reuse_the_previous_artifact_identity() -> None:
    original = build_run("2026.10.22.2", steps=3, data_version="2026.10.22.2", **SOURCE_URIS)
    longer = build_run("2026.10.22.2", steps=25, data_version="2026.10.22.2", **SOURCE_URIS)
    changed_data = build_run("2026.10.22.2", steps=3, data_version="2026.10.22.3", **SOURCE_URIS)

    assert len({original.fingerprint(), longer.fingerprint(), changed_data.fingerprint()}) == 3
