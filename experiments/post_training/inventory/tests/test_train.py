# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin.execution.lazy import StepContext
from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training.inventory.train import build_run, main


def test_missing_judge_credential_rejects_inventory_run_before_iris_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "offline-test-key")

    def unexpected_connection(**_kwargs):
        pytest.fail("Missing judge credential reached the external Iris connection")

    monkeypatch.setattr("marin.rl.cli.open_iris_client", unexpected_connection)
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.23.1",
            "--train-source",
            "s3://test/training.parquet",
            "--validation-source",
            "s3://test/validation.parquet",
            "--steps",
            "3",
            "--run",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, KeyError)
    assert result.exception.args == ("TOGETHER_API_KEY",)


@pytest.mark.timeout(180)
def test_three_update_run_composes_native_evaluation_and_terminal_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the runtime boundary: a three-update run must retain initial/final
    # native-task evaluation and an aligned terminal checkpoint/export schedule.
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "offline-test-key")
    run = build_run(
        "2026.10.22.2",
        steps=3,
        training_uri="s3://test/training.parquet",
        validation_uri="s3://test/validation.parquet",
    )
    config = run.build_config(StepContext.for_fingerprint(run.runtime_args, run.deps))
    path = tmp_path / "launch.yaml"
    path.write_text(config.launch_config_yaml)
    script = """
import json
import sys
from omegaconf import OmegaConf
from shellbox.image import RegistryImage
from shellbox.machine import Backend, CommandSemantics, MachineSpec
from taskcompendium.models import EnvironmentRequirements
from marinskyrl.task_machine_factories import task_machine_factories
from cloud.iris.launch_config import load_launch_config

launch = load_launch_config(sys.argv[1])
trainer = launch.skyrl.trainer
# Image-backed heldout graders must reach a backend supporting RegistryImage.
# Construct only the remote backend so this pure admission check is host independent.
factories = task_machine_factories(OmegaConf.create({
    "daytona": launch.skyrl.environment.task_sessions.daytona,
}))
image = "fixture@sha256:" + "a" * 64
factories[Backend.DAYTONA].check_environment(
    EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=image),
    MachineSpec(source=RegistryImage(image)),
)
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
    "image_grader_admitted": True,
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
        "updates": 3,
        "evaluation_interval": 3,
        "checkpoint_interval": 3,
        "hf_interval": 3,
        "export_hf": True,
        "all_heldout_prompts": True,
        "image_grader_admitted": True,
    }
