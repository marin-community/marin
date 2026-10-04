# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import yaml
from click.testing import CliRunner
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.experiment.cli import graph_handles
from marin.training.training import LevanterCheckpoint
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.launch import main, spike_workflow


def test_spike_retains_frozen_development_and_checks_rewards_before_policy_allocation():
    seed = ArtifactStep.adopt("documents/test-russell-seed", "2026.10.04", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/test-russell-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/test-russell-wheels", "2026.10.04", "/tmp/wheels")
    terminals = spike_workflow(
        seed,
        parent,
        "smoke",
        "2026.10.04",
        "test-relay",
        "test-image",
        RuntimeBundle("/tmp/manifest.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64),
        {"backend": "qemu", "qemu": {}},
        wheels,
    )
    handles = {handle.name: handle for handle in graph_handles(list(terminals.values()))}
    baseline = handles["evals/russell-rsi-parent-development"]
    adaptive = handles["documents/russell-rsi-adaptive-round-1"]
    calibration = handles["evals/russell-rsi-train-calibration-development"]
    trained = handles["checkpoints/russell-rsi-smoke"]
    assert baseline in adaptive.deps
    assert calibration in trained.deps
    launch = yaml.safe_load(json.loads(trained.fingerprint_payload())["launch_config_yaml"])
    assert launch["inputs"]["train_data"][0]["identity"] == artifact_identity(adaptive)
    assert launch["inputs"]["validation_data"][0]["identity"] == artifact_identity(seed)
    calibration_config = json.loads(calibration.fingerprint_payload())
    assert calibration_config["require_reward_variation"] is True
    assert calibration_config["samples_per_task"] == 4


def test_spike_run_rejects_missing_generation_token_before_building(monkeypatch):
    monkeypatch.delenv("GLM_API_TOKEN", raising=False)
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.04",
            "--stage",
            "spike",
            "--scale",
            "smoke",
            "--data-name",
            "documents/test-seed",
            "--data-version",
            "2026.10.04",
            "--data-uri",
            "/tmp/seed",
            "--model-uri",
            "/tmp/model",
            "--run",
        ],
    )
    assert result.exit_code == 2
    assert "requires GLM_API_TOKEN before any GPU work" in result.output
