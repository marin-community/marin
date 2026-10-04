# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.experiment import cli as experiment_cli
from marin.experiment.cli import graph_handles
from marin.training.training import LevanterCheckpoint
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.launch import (
    MODEL,
    MODEL_REVISION,
    development_step,
    main,
    repair_spike_workflow,
    spike_workflow,
)


def test_spike_retains_frozen_development_and_checks_rewards_before_policy_allocation():
    seed = ArtifactStep.adopt("documents/test-russell-seed", "2026.10.04", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/test-russell-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/test-russell-wheels", "2026.10.04", "/tmp/wheels")
    sources = ArtifactStep.adopt("documents/test-russell-sources", "2026.10.04", "/tmp/sources")
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
        sources,
        "0" * 64,
    )
    handles = {handle.name: handle for handle in graph_handles(list(terminals.values()))}
    baseline = handles["evals/russell-rsi-parent-development"]
    adaptive = handles["documents/russell-rsi-adaptive-round-1"]
    calibration = handles["evals/russell-rsi-train-calibration-development"]
    trained = handles["checkpoints/russell-rsi-smoke"]
    assert baseline in adaptive.deps
    assert sources in adaptive.deps
    assert sources not in baseline.deps
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


def test_smoke_and_pilot_share_admission_without_evaluation_collisions():
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.2", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/pinned-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/wheels", "2026.10.04.2", "/tmp/wheels")
    sources = ArtifactStep.adopt("documents/sources", "2026.10.04.2", "/tmp/sources")
    terminals = {
        scale: spike_workflow(
            seed,
            parent,
            scale,
            "2026.10.04.2",
            "test-relay",
            "test-image",
            RuntimeBundle("/tmp/manifest.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64),
            {"backend": "qemu", "qemu": {}},
            wheels,
            sources,
            "0" * 64,
        )
        for scale in ("smoke", "pilot")
    }
    handles = {
        scale: {handle.name: handle for handle in graph_handles(list(outputs.values()))}
        for scale, outputs in terminals.items()
    }
    smoke = handles["smoke"]
    pilot = handles["pilot"]
    assert {name for name in smoke if name.startswith("evals/")} == {
        "evals/russell-rsi-parent-development",
        "evals/russell-rsi-train-calibration-development",
        terminals["smoke"]["reload"].name,
    }
    for name in (
        "evals/russell-rsi-parent-development",
        "documents/russell-rsi-adaptive-round-1",
        "evals/russell-rsi-train-calibration-development",
    ):
        assert artifact_identity(smoke[name]) == artifact_identity(pilot[name])
    assert terminals["pilot"]["development"].name == "evals/russell-rsi-candidate-pilot-development"
    assert artifact_identity(smoke["checkpoints/russell-rsi-smoke"]) != artifact_identity(
        pilot["checkpoints/russell-rsi-pilot"]
    )
    assert terminals["smoke"]["reload"].name not in pilot
    assert all(handle.name not in smoke for handle in terminals["pilot"].values())
    candidate = terminals["pilot"]["development"]
    candidate_public = terminals["pilot"]["candidate-coding-subset"]
    pilot_reload = next(handle for handle in candidate_public.deps if "mmlu-smoke" in handle.name)
    assert pilot_reload in candidate.deps


def test_parent_development_plan_reuses_spike_baseline_without_training(monkeypatch):
    captured = []
    monkeypatch.setattr(experiment_cli, "_print_plan", captured.extend)
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.04.2",
            "--stage",
            "parent-development",
            "--scale",
            "smoke",
            "--data-name",
            "documents/frozen-seed",
            "--data-version",
            "2026.10.04.2",
            "--data-uri",
            "/tmp/seed",
            "--model-uri",
            "/tmp/model",
            "--machine-config-json",
            json.dumps(
                {
                    "backend": "qemu",
                    "runtime_bundle": {
                        "manifest_uri": "/tmp/manifest.json",
                        "manifest_sha256": "0" * 64,
                        "archive_uri": "/tmp/runtime.tar.gz",
                        "archive_sha256": "0" * 64,
                    },
                }
            ),
        ],
    )
    assert result.exit_code == 0, result.output
    assert [handle.name for handle in graph_handles(captured)] == [
        "documents/frozen-seed",
        "checkpoints/russell-sft-parent",
        "evals/russell-rsi-parent-development",
    ]
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.2", "/tmp/seed")
    parent = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        "/tmp/model",
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    terminals = spike_workflow(
        seed,
        parent,
        "smoke",
        "2026.10.04.2",
        "test-relay",
        "test-image",
        RuntimeBundle("/tmp/manifest.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64),
        {"backend": "qemu", "qemu": {}},
        ArtifactStep.adopt("documents/wheels", "2026.10.04.2", "/tmp/wheels"),
        ArtifactStep.adopt("documents/sources", "2026.10.04.2", "/tmp/sources"),
        "0" * 64,
    )
    baseline = next(
        handle
        for handle in graph_handles(list(terminals.values()))
        if handle.name == "evals/russell-rsi-parent-development"
    )
    assert artifact_identity(captured[0]) == artifact_identity(baseline)


def test_repair_spike_uses_qualified_union_without_running_failed_adaptive_round():
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.1", "/tmp/seed")
    parent = ArtifactStep.adopt("checkpoints/pinned-parent", "2026.09.21", "/tmp/model", kind=LevanterCheckpoint)
    wheels = ArtifactStep.adopt("documents/wheels", "2026.10.04.2", "/tmp/wheels")
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    terminals = {
        scale: repair_spike_workflow(
            seed,
            parent,
            scale,
            "2026.10.04.2",
            relay_job="test-relay",
            image="test-image",
            runtime_bundle=runtime,
            machine_config={"backend": "qemu"},
            wheels=wheels,
            manifest_uri="/tmp/sealed-manifest.json",
            manifest_sha256="1" * 64,
        )
        for scale in ("smoke", "pilot")
    }
    graphs = {
        scale: {handle.name: handle for handle in graph_handles(list(outputs.values()))}
        for scale, outputs in terminals.items()
    }
    smoke, pilot = graphs["smoke"], graphs["pilot"]
    assert "documents/russell-rsi-adaptive-round-1" not in smoke
    union = smoke["documents/russell-rsi-qualified-union-1"]
    repaired = smoke["documents/russell-rsi-repair-1"]
    evidence = smoke["documents/russell-rsi-round-1-evidence"]
    baseline = smoke["evals/russell-rsi-parent-development"]
    union_config = json.loads(union.fingerprint_payload())
    assert union_config["parent_development_identity"] == artifact_identity(baseline)
    assert union_config["original_manifest_sha256"] == "1" * 64
    assert evidence in repaired.deps and repaired in union.deps
    trained = smoke["checkpoints/russell-rsi-repair-1-smoke"]
    launch = yaml.safe_load(json.loads(trained.fingerprint_payload())["launch_config_yaml"])
    assert launch["inputs"]["train_data"][0]["identity"] == artifact_identity(union)
    assert launch["inputs"]["validation_data"][0]["identity"] == artifact_identity(seed)
    calibration = smoke["evals/russell-rsi-repair-1-train-calibration-development"]
    assert calibration in trained.deps
    for handle in (baseline, repaired, union, calibration):
        assert artifact_identity(handle) == artifact_identity(pilot[handle.name])
    assert terminals["smoke"].keys() == {"reload"}
    assert terminals["pilot"]["development"].name == "evals/russell-rsi-candidate-repair-1-pilot-development"


@pytest.mark.parametrize("change", ["bytes", "parent"])
def test_repair_cli_rejects_changed_evidence_before_building(change, tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(experiment_cli, "_print_plan", captured.extend)
    seed = ArtifactStep.adopt("documents/frozen-seed", "2026.10.04.1", "/tmp/seed")
    parent = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        "/tmp/model",
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    runtime = RuntimeBundle("/tmp/runtime.json", "0" * 64, "/tmp/runtime.tar.gz", "0" * 64)
    baseline = development_step(seed, parent, "2026.10.04.2", runtime, "parent")
    identity = artifact_identity(baseline) if change == "bytes" else "different-parent"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs": {"parent_development_identity": identity}}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if change == "bytes":
        manifest.write_text(manifest.read_text() + "\n")
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.04.2",
            "--stage",
            "repair-spike",
            "--scale",
            "smoke",
            "--data-name",
            seed.name,
            "--data-version",
            seed.version,
            "--data-uri",
            "/tmp/seed",
            "--model-uri",
            "/tmp/model",
            "--relay-job",
            "test-relay",
            "--task-image",
            "test-image",
            "--dependency-wheels-uri",
            "/tmp/wheels",
            "--repair-manifest-uri",
            str(manifest),
            "--repair-manifest-sha256",
            digest,
            "--machine-config-json",
            json.dumps(
                {
                    "backend": "qemu",
                    "runtime_bundle": {
                        "manifest_uri": runtime.manifest_uri,
                        "manifest_sha256": runtime.manifest_sha256,
                        "archive_uri": runtime.archive_uri,
                        "archive_sha256": runtime.archive_sha256,
                    },
                }
            ),
        ],
    )
    assert result.exit_code != 0
    error = str(result.exception) + result.output
    assert ("digest mismatch" if change == "bytes" else "parent-development identity") in error
    assert captured == []
