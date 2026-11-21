"""Training driver input resolution and checkpoint export."""

import json

import pytest
from omegaconf import OmegaConf

from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner


def test_checkpoint_export_entrypoint_bypasses_rollout_environment(monkeypatch, tmp_path):
    launch_config = OmegaConf.create(
        {
            "run": {"mode": "checkpoint_export"},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.checkpoint_export"},
            "skyrl": {"trainer": {"policy": {"model": {"path": "/tmp/policy"}}}},
        }
    )
    cfg = LocalRLConfig(
        job_name="checkpoint-export",
        model_path="Qwen/Qwen3-8B",
        train_data=[OmegaConf.create({"source": "unused-during-export"})],
        resolved_config_uri=(tmp_path / "resolved.json").as_uri(),
        gpus=4,
        launch_config=launch_config,
    )
    runner = LocalRLRunner(cfg)
    invocation = {}

    monkeypatch.setattr(
        runner,
        "_setup_environment",
        lambda _args: pytest.fail("checkpoint export must not configure the rollout runtime"),
    )
    monkeypatch.setattr(runner, "_run_skyrl", lambda config: invocation.update(run=config) or 0)

    assert runner.run() == 0
    assert invocation == {"run": launch_config}
    assert json.loads((tmp_path / "resolved.json").read_text()) == {
        "config": OmegaConf.to_container(launch_config, resolve=True),
        "train_data_sources": [{"source": "unused-during-export"}],
        "val_data_sources": [],
    }
