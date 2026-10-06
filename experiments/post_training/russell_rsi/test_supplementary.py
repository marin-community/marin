# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pytest

from experiments.post_training.russell_rsi.launch import MODEL, MODEL_REVISION
from experiments.post_training.russell_rsi.launch_supplementary import (
    AcceptanceResultConfig,
    SelectedCheckpoint,
    seal_acceptance_result,
    selected_checkpoint,
)


@pytest.mark.parametrize("kind", ["skyrl", "sft"])
def test_selected_export_must_match_the_promoted_checkpoint_before_panel_execution(tmp_path, kind):
    parent = {"checkpoint_identity": "parent", "development": [0.5, 0.5], "retention": 0.5}
    candidate = {**parent, "checkpoint_identity": "checkpoints/candidate@v1:abcd", "development": [0.6, 0.5]}
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"parent": parent, "selected": candidate}))
    record = tmp_path / "checkpoint.json"
    exported = "s3://unit/candidate/export"
    payload = {
        "name": "checkpoints/candidate",
        "version": "v1",
        "fingerprint": "abcd",
        "result_type": "marin.rl.skyrl.SkyRLRun" if kind == "skyrl" else "marin.training.training.LevanterCheckpoint",
        "source": exported if kind == "sft" else None,
        "result": {"hf_model_uri": exported, "tokenizer_uri": MODEL, "tokenizer_revision": MODEL_REVISION},
    }
    record.write_text(json.dumps(payload))
    config = {
        "parent": {"artifact_identity": "parent"},
        "decision_uri": str(decision),
        "decision_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
        "checkpoint_record_uri": str(record),
        "checkpoint_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
    }
    assert selected_checkpoint(config).export_uri == exported
    missing = json.loads(record.read_text())
    if kind == "skyrl":
        missing["result"]["hf_model_uri"] = None
    else:
        missing["source"] = None
    record.write_text(json.dumps(missing))
    config["checkpoint_record_sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="completed HF export"):
        selected_checkpoint(config)
    record.write_text(json.dumps(payload))
    stale = json.loads(record.read_text())
    stale["fingerprint"] = "different"
    record.write_text(json.dumps(stale))
    config["checkpoint_record_sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not identify the selected checkpoint"):
        selected_checkpoint(config)
    decision.write_text(json.dumps({"parent": parent, "selected": {**candidate, "retention": 0.0}}))
    config["decision_sha256"] = hashlib.sha256(decision.read_bytes()).hexdigest()
    record.unlink()
    with pytest.raises(ValueError, match="parent promotion gate"):
        selected_checkpoint(config)


@pytest.mark.parametrize("selected,passed", [([1, 0, 1, 1], True), ([1, 0, 1, 0], False)])
def test_acceptance_reports_all_matched_gains_and_losses_without_reselection(tmp_path, selected, passed):
    paths = []
    for label, rewards in (("parent", [1, 1, 0, 0]), ("candidate", selected)):
        path = tmp_path / label
        path.mkdir()
        (path / "failure_summary.json").write_text(
            json.dumps(
                {
                    "model_identity": label,
                    "tasks_identity": "panel",
                    "count": 4,
                    "samples_per_task": 1,
                    "task_rewards": {str(index): [reward] for index, reward in enumerate(rewards)},
                }
            )
        )
        paths.append(str(path))
    config = AcceptanceResultConfig(
        (paths[0], paths[1]),
        ("parent", "candidate"),
        "panel",
        ("0", "1", "2", "3"),
        SelectedCheckpoint("/unused/export", "selected-before-panel", "a" * 64, "b" * 64),
        str(tmp_path / "report"),
    )
    seal_acceptance_result(config)
    report = tmp_path / "report/acceptance-result.json"
    original = report.read_bytes()
    result = json.loads(original)
    assert result["acceptance_passed"] is passed
    assert result["paired_net_gain"] == sum(selected) - 2
    assert result["lost"] == ["1"]
    assert result["gained"] == (["2", "3"] if passed else ["2"])
    assert result["selected_source"]["source_identity"] == "selected-before-panel"
    seal_acceptance_result(config)
    assert report.read_bytes() == original
    path = tmp_path / "candidate/failure_summary.json"
    summary = json.loads(path.read_text())
    summary["task_rewards"]["2"] = []
    path.write_text(json.dumps(summary))
    incomplete = replace(config, output_path=str(tmp_path / "incomplete"))
    with pytest.raises(ValueError, match="one valid grade"):
        seal_acceptance_result(incomplete)
    assert not (tmp_path / "incomplete").exists()
