# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.experiment.cli import graph_handles
from marin.rl.skyrl import IrisSkyRLExecution, SkyRLRun
from marin.training.training import LevanterCheckpoint

from experiments.post_training.russell_rsi.bootstrap_loop import CheckpointScore
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingPanel
from experiments.post_training.russell_rsi.launch import SamplingMode, Scale, train_step
from experiments.post_training.russell_rsi.launch_dose_comparison import (
    DoseSelectionConfig,
    SavedExportConfig,
    dose_workflow,
    export_saved_four,
    require_dose_source_launch,
    require_duration_only_launch,
    seal_dose_selection,
    selected_dose,
)


@pytest.mark.parametrize(
    "coding,retention,expected",
    [
        ((0.6, 0.5), 0.8, "eight"),
        ((0.5, 0.6), 0.8, "eight"),
        ((0.5, 0.5), 0.8, "four"),
        ((0.6, 0.4), 0.8, "four"),
        ((0.6, 0.6), 0.7, "four"),
    ],
)
def test_checkpoint_selection_requires_retention_and_unopposed_coding_gain(coding, retention, expected):
    four = CheckpointScore("four", (0.5, 0.5), 0.8)
    eight = CheckpointScore("eight", coding, retention)
    assert selected_dose(four, eight).checkpoint_identity == expected


def test_every_dose_evaluation_waits_for_both_checkpoint_exports():
    def value(name):
        return {"name": name, "version": "2026.10.05", "uri": f"s3://test/{name}", "identity_config": {}}

    config = {
        "version": "2026.10.05",
        "parent": value("parent"),
        "bank": value("bank"),
        "retention": value("retention"),
        "machine_config": {},
        "runtime_bundle": {
            "manifest_uri": "s3://test/manifest",
            "manifest_sha256": "a" * 64,
            "archive_uri": "s3://test/archive",
            "archive_sha256": "b" * 64,
        },
        "retention_count": 16,
        "parent_score": {"checkpoint_identity": "parent", "development": (0.2, 0.3), "retention": 0.5},
        "qualification": {"files": {}},
        "qualified_four_export_uri": "s3://test/export4",
        "source_reload_identity": "reload-original",
        "retention_task_ids": [str(index) for index in range(16)],
    }
    parent = ArtifactStep.adopt("parent", "2026.10.05", "s3://test/parent", kind=LevanterCheckpoint, config={})
    bank = ArtifactStep.adopt("bank", "2026.10.05", "s3://test/bank", kind=Artifact, config={})
    schedule = {
        "parent_identity": artifact_identity(parent),
        "bank_identity": artifact_identity(bank),
        "source_schedule_sha256": "a" * 64,
    }
    outputs = dose_workflow(config, SimpleNamespace(task_bank=()), schedule, CodingPanel((), {}))
    for name in ("reload-4", "reload-8", "coding-4", "coding-8", "retention-4", "retention-8"):
        ancestors = graph_handles([outputs[name]])
        assert outputs["rl"] in ancestors
        assert outputs["export-four"] in ancestors
    assert artifact_identity(outputs["reload-4"]) != artifact_identity(outputs["reload-8"])
    assert artifact_identity(outputs["coding-4"]) != artifact_identity(outputs["coding-8"])


def test_saved_four_export_reads_immutable_request_and_waits_for_completion(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    checkpoint = root / "global_step_4"
    checkpoint.mkdir(parents=True)
    request_file = checkpoint / "hf_export_request.json"
    export = str(tmp_path / "four-export")
    request = {"step": 4, "checkpoint_path": str(checkpoint), "export_path": export, "status": "pending"}
    request_file.write_text(json.dumps(request))
    launch = tmp_path / "resolved-launch.yaml"
    launch_config = {"schema_version": 1, "runtime": {"launcher_commit": "pinned"}}
    launch.write_text(yaml.safe_dump({"config": launch_config, "train_data_sources": [], "val_data_sources": []}))
    trained = SkyRLRun(
        path=str(tmp_path / "terminal.json"),
        hf_model_uri=str(tmp_path / "eight-export"),
        global_step=8,
        tokenizer_uri="tokenizer",
        tokenizer_revision="pin",
        checkpoint_root=str(root),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(tmp_path / "terminal.json"),
        iris_job_id="job",
    )
    submitted = []

    def completed_export(argv, *, check):
        submitted.append(argv)
        assert argv[argv.index("--request") + 1] == str(checkpoint)
        assert "--no-wait" not in argv
        assert yaml.safe_load(Path(argv[argv.index("--launch-config") + 1]).read_text()) == launch_config
        # The pinned export subprocess changes to its installed package directory.
        with monkeypatch.context() as exporter:
            exporter.chdir(tmp_path)
            for option in ("--cluster-config", "--parent-cluster-config"):
                assert yaml.safe_load(Path(argv[argv.index(option) + 1]).read_text())
        request_file.write_text(json.dumps({**request, "status": "complete", "last_exit_code": 0}))

    monkeypatch.setattr("experiments.post_training.russell_rsi.launch_dose_comparison.subprocess.run", completed_export)

    def exported_policy(argv, *, text):
        assert argv[-2:] == [export, "4"]
        return str(Path(export) / "global_step_4/policy") + "\n"

    monkeypatch.setattr(
        "experiments.post_training.russell_rsi.launch_dose_comparison.subprocess.check_output", exported_policy
    )
    result = export_saved_four(SavedExportConfig(trained, str(launch), str(tmp_path / "export-evidence")))
    assert result.global_step == 4
    assert result.hf_model_uri == str(Path(export) / "global_step_4/policy")
    assert trained.global_step == 8
    assert submitted


@pytest.mark.parametrize("missing", [False, True])
def test_dose_selection_rejects_missing_retention_tasks_before_choice(tmp_path, missing):
    coding_paths = []
    retention_paths = []
    for step in (4, 8):
        coding = tmp_path / f"coding-{step}"
        retained = tmp_path / f"retention-{step}"
        coding.mkdir()
        retained.mkdir()
        (coding / "coding-evidence.json").write_text(
            json.dumps(
                {
                    "model_identity": str(step),
                    "scores": {"humanevalplus": 0.5 if step == 4 else 0.6, "mbppplus": 0.5},
                }
            )
        )
        rewards = {"one": [1], "two": [0]}
        if missing and step == 8:
            rewards.pop("two")
        (retained / "failure_summary.json").write_text(
            json.dumps(
                {
                    "model_identity": str(step),
                    "tasks_identity": "retention",
                    "count": 2,
                    "task_rewards": rewards,
                }
            )
        )
        coding_paths.append(str(coding))
        retention_paths.append(str(retained))
    output = tmp_path / "selection"
    config = DoseSelectionConfig(
        tuple(coding_paths),
        tuple(retention_paths),
        ("4", "8"),
        "retention",
        ("one", "two"),
        CheckpointScore("parent", (0.2, 0.3), 0.5),
        str(output),
    )
    if missing:
        with pytest.raises(ValueError, match="same complete evaluation panels"):
            seal_dose_selection(config)
        assert not (output / "dose-selection.json").exists()
    else:
        seal_dose_selection(config)
        selected = json.loads((output / "dose-selection.json").read_text())
        assert selected["selected"]["checkpoint_identity"] == "8"
        assert selected["parent"] == {"checkpoint_identity": "parent", "development": [0.2, 0.3], "retention": 0.5}


@pytest.mark.parametrize("change", ["context", "optimizer", "allocation"])
def test_qualified_source_rejects_recipe_or_resource_changes(change):
    data = ArtifactStep.adopt("data", "2026.10.05", "/tmp/data", config={})
    parent = ArtifactStep.adopt("parent", "2026.10.05", "/tmp/parent", kind=LevanterCheckpoint, config={})
    reference = train_step(
        data, parent, "pilot", "2026.10.05", machine_config={}, sampling_mode=SamplingMode.CALIBRATED_REPLAY
    )
    expected = yaml.safe_load(json.loads(reference.fingerprint_payload())["launch_config_yaml"])
    execution = next(item for item in reference.runtime_args.values() if isinstance(item, IrisSkyRLExecution))
    source = deepcopy(expected)
    source["iris"]["allocation"].update(cpu=execution.cpu, memory=execution.memory, disk=execution.disk)
    source["iris"]["cluster"] = execution.cluster
    require_dose_source_launch(source, expected, execution, artifact_identity(parent))
    if change == "context":
        source["skyrl"]["context_budget"]["max_prompt_tokens"] -= 1
    elif change == "optimizer":
        source["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] *= 2
    else:
        source["iris"]["allocation"]["num_nodes"] -= 1
    with pytest.raises(ValueError, match="canonical dose source recipe"):
        require_dose_source_launch(source, expected, execution, artifact_identity(parent))


def test_actual_dose_launch_rejects_an_extra_optimizer_change():
    data = ArtifactStep.adopt("data", "2026.10.05", "/tmp/data", config={})
    parent = ArtifactStep.adopt("parent", "2026.10.05", "/tmp/parent", kind=LevanterCheckpoint, config={})
    four = train_step(data, parent, "pilot", "2026.10.05", sampling_mode=SamplingMode.CALIBRATED_REPLAY)
    eight = train_step(
        data,
        parent,
        "dose",
        "2026.10.05",
        sampling_mode=SamplingMode.CALIBRATED_REPLAY,
        bounded_scale=Scale(8, 7),
        checkpoint_interval=4,
    )
    requested_four = yaml.safe_load(json.loads(four.fingerprint_payload())["launch_config_yaml"])
    requested_eight = yaml.safe_load(json.loads(eight.fingerprint_payload())["launch_config_yaml"])
    require_duration_only_launch(requested_four, requested_eight)
    requested_eight["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] *= 2
    with pytest.raises(ValueError, match="changes settings beyond"):
        require_duration_only_launch(requested_four, requested_eight)
