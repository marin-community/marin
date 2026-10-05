# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict

import pytest
from marin.execution.lazy import ArtifactStep, artifact_identity

from experiments.post_training.russell_rsi import launch_bootstrap_loop
from experiments.post_training.russell_rsi.bootstrap_loop import CheckpointScore, LoopState, QualifiedTask
from experiments.post_training.russell_rsi.feedback import SKILL_DESCRIPTIONS, CodingSkill
from experiments.post_training.russell_rsi.launch import ReviewedConstructionInputs
from experiments.post_training.russell_rsi.sources import compact_json_sha256


def _capabilities(labels: tuple[str, ...]) -> dict:
    return {"skills": [{"label": label, "description": SKILL_DESCRIPTIONS[CodingSkill(label)]} for label in labels]}


def _execute_callback(
    tmp_path,
    monkeypatch,
    reviewed_record,
    bank_feedback=None,
    *,
    raw_identity=None,
    raw_sha=None,
    reviewed_pin=None,
    raw_record_override=None,
    review_record_override=None,
    bank_missing=False,
    review_missing=False,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_record = raw_record_override or _capabilities(("types", "boundaries", "state"))
    raw_bytes = json.dumps(raw_record).encode()
    (raw_dir / "capabilities.json").write_bytes(raw_bytes)
    reviewed_dir = tmp_path / "reviewed"
    reviewed_dir.mkdir()
    reviewed_bytes = json.dumps(reviewed_record).encode()
    (reviewed_dir / "capabilities.json").write_bytes(reviewed_bytes)
    reviewed_sha = hashlib.sha256(reviewed_bytes).hexdigest()
    actual_raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    review_record = review_record_override or {
        "source_capabilities_sha256": actual_raw_sha,
        "reviewed_capabilities_sha256": reviewed_sha,
    }
    review_record_path = tmp_path / "review-record.json"
    review_record_bytes = json.dumps(review_record).encode()
    review_record_path.write_bytes(review_record_bytes)
    raw_feedback = ArtifactStep.adopt("documents/raw-feedback", "2026.10.04", str(raw_dir))

    tasks = (QualifiedTask("task", "task-hash", "admission", "source", "types", "contract"),)
    state = LoopState(
        CheckpointScore("parent", (1.0, 1.0), 1.0),
        CheckpointScore("working", (1.0, 1.0), 1.0),
        CheckpointScore("champion", (1.0, 1.0), 1.0),
        tasks,
        completed_pilots=1,
    )
    bank_uri = tmp_path / "bank"
    bank_uri.mkdir()
    bank_bytes = json.dumps(
        {"tasks": [asdict(task) for task in tasks], "feedback_identity": bank_feedback or reviewed_sha}
    ).encode()
    (bank_uri / "bank.json").write_bytes(bank_bytes)
    bank_sha = hashlib.sha256(bank_bytes).hexdigest()
    entry = {
        "raw_feedback_identity": raw_identity or artifact_identity(raw_feedback),
        "raw_capabilities_sha256": raw_sha or actual_raw_sha,
        "reviewed_feedback": {
            "name": "documents/reviewed-feedback",
            "version": "2026.10.04",
            "uri": str(reviewed_dir),
            "identity_config": {"review": "approved"},
            "capabilities_sha256": reviewed_pin or reviewed_sha,
            "review_record_uri": str(review_record_path),
            "review_record_sha256": hashlib.sha256(review_record_bytes).hexdigest(),
        },
        "prior_bank_sha256": compact_json_sha256({"tasks": [asdict(task) for task in tasks]}),
        "response_cap": 24,
        "bank": (
            None
            if bank_missing
            else {
                "name": "documents/reviewed-bank",
                "version": "2026.10.04",
                "uri": str(bank_uri),
                "identity_config": {},
                "bank_sha256": bank_sha,
            }
        ),
    }
    config = {
        "seed_bank": {"name": "documents/seed", "version": "2026.10.04", "uri": str(bank_uri), "identity_config": {}},
        "parent": {"name": "checkpoints/parent", "version": "2026.10.04", "uri": "/tmp/parent", "identity_config": {}},
        "retention": {
            "name": "documents/retention",
            "version": "2026.10.04",
            "uri": "/tmp/retention",
            "identity_config": {},
        },
        "panel_uri": str(tmp_path / "panel.json"),
        "panel_sha256": "panel-hash",
        "reviewed_feedback": {} if review_missing else {"1": entry},
        "heldout_manifest_uri": str(tmp_path / "heldout.json"),
        "heldout_manifest_sha256": "heldout-sha",
        "parent_coding_evidence_uri": str(tmp_path / "coding.json"),
        "parent_coding_evidence_sha256": "coding-sha",
        "parent_retention_evidence_uri": str(tmp_path / "retention.json"),
        "parent_retention_evidence_sha256": "retention-sha",
        "version": "2026.10.04",
        "runtime_bundle": {
            "manifest_uri": "/tmp/runtime.json",
            "manifest_sha256": "0" * 64,
            "archive_uri": "/tmp/runtime.tar.gz",
            "archive_sha256": "0" * 64,
        },
        "machine_config": {"backend": "qemu"},
        "relay_job": "relay",
        "manifest_prefix": str(tmp_path / "manifests"),
    }
    panel = json.dumps({"items": [], "protocols": {}}).encode()
    (tmp_path / "panel.json").write_bytes(panel)
    config["panel_sha256"] = hashlib.sha256(panel).hexdigest()
    result = []

    def run_loop(
        seed_bank,
        parent,
        retention,
        panel,
        heldout_manifest_uri,
        heldout_manifest_sha256,
        parent_coding_evidence_uri,
        parent_coding_evidence_sha256,
        parent_retention_evidence_uri,
        parent_retention_evidence_sha256,
        version,
        runtime_bundle,
        machine_config,
        relay_job,
        manifest_directory,
        next_construction_inputs,
        initial_calibration=None,
        predecessor=None,
    ):
        result.append(next_construction_inputs(artifact_identity(raw_feedback), raw_bytes, state, 24))

    monkeypatch.setattr(launch_bootstrap_loop, "run_bootstrap_loop", run_loop)
    launch_bootstrap_loop.execute_loop(config)
    return raw_feedback, state, entry, bank_sha, result[0]


def test_execute_loop_adopts_reviewed_capabilities_and_bank(tmp_path, monkeypatch):
    reviewed = _capabilities(("types",))
    raw, state, entry, _, inputs = _execute_callback(tmp_path, monkeypatch, reviewed)
    assert artifact_identity(raw) == entry["raw_feedback_identity"]
    assert compact_json_sha256({"tasks": [asdict(task) for task in state.bank]}) == entry["prior_bank_sha256"]
    assert isinstance(inputs, ReviewedConstructionInputs)
    assert inputs.bank is not None
    assert artifact_identity(inputs.feedback) != artifact_identity(raw)
    assert inputs.feedback.adopt_config["raw_feedback_identity"] == artifact_identity(raw)
    assert inputs.feedback.adopt_config["review_record_sha256"] == entry["reviewed_feedback"]["review_record_sha256"]
    assert inputs.bank.adopt_config["feedback_identity"] == artifact_identity(inputs.feedback)
    assert inputs.bank.adopt_config["capabilities_sha256"] == entry["reviewed_feedback"]["capabilities_sha256"]
    assert inputs.capabilities_bytes == json.dumps(reviewed).encode()


@pytest.mark.parametrize(
    "reviewed,bank_feedback,error",
    [
        (
            {
                "skills": [
                    {"label": "types", "description": SKILL_DESCRIPTIONS[CodingSkill.TYPES], "evidence": "private"}
                ]
            },
            None,
            "canonical schema",
        ),
        ({"skills": _capabilities(("types",))["skills"], "private": "evidence"}, None, "canonical schema"),
        (_capabilities(("types", "types")), None, "duplicate labels"),
        (_capabilities(("types", "error_handling")), None, "adds a label"),
        ({"skills": [{"label": "types", "description": "changed"}]}, None, "canonical description"),
        (_capabilities(("types",)), "raw-sha", "does not cite the reviewed feedback"),
    ],
)
def test_execute_loop_rejects_unreviewed_or_changed_feedback(tmp_path, monkeypatch, reviewed, bank_feedback, error):
    with pytest.raises(ValueError, match=error):
        _execute_callback(tmp_path, monkeypatch, reviewed, bank_feedback=bank_feedback)


@pytest.mark.parametrize("wrong_field", ["raw_feedback_identity", "raw_capabilities_sha256"])
def test_execute_loop_rejects_raw_feedback_drift(tmp_path, monkeypatch, wrong_field):
    reviewed = _capabilities(("types",))
    wrong = "wrong-pin"
    with pytest.raises(ValueError, match=r"raw coding feedback|raw capability bytes"):
        _execute_callback(
            tmp_path,
            monkeypatch,
            reviewed,
            **{"raw_identity" if wrong_field == "raw_feedback_identity" else "raw_sha": wrong},
        )


def test_execute_loop_rejects_noncanonical_raw_description(tmp_path, monkeypatch):
    raw = _capabilities(("types", "boundaries", "state"))
    raw["skills"][0]["description"] = "Unreviewed text"
    with pytest.raises(ValueError, match="changes a canonical description"):
        _execute_callback(tmp_path, monkeypatch, _capabilities(("types",)), raw_record_override=raw)


def test_execute_loop_rejects_reviewed_bytes_that_do_not_match_the_pin(tmp_path, monkeypatch):
    reviewed = _capabilities(("types",))
    with pytest.raises(ValueError, match="digest mismatch"):
        _execute_callback(tmp_path, monkeypatch, reviewed, reviewed_pin="0" * 64)


def test_execute_loop_requires_review_record_to_bind_both_capability_files(tmp_path, monkeypatch):
    reviewed = _capabilities(("types",))
    review_record = {"source_capabilities_sha256": "wrong", "reviewed_capabilities_sha256": "wrong"}
    with pytest.raises(ValueError, match="Review record does not identify"):
        _execute_callback(tmp_path, monkeypatch, reviewed, review_record_override=review_record)


def test_review_without_bank_returns_distinct_construction_pending_state(tmp_path, monkeypatch):
    reviewed = _capabilities(("types",))
    _raw, state, _entry, _bank_sha, inputs = _execute_callback(tmp_path, monkeypatch, reviewed, bank_missing=True)
    assert state.completed_pilots == 1
    assert isinstance(inputs, ReviewedConstructionInputs)
    assert inputs.bank is None


def test_empty_review_is_a_reviewed_input(tmp_path, monkeypatch):
    _raw, _state, _entry, _bank_sha, inputs = _execute_callback(
        tmp_path, monkeypatch, _capabilities(()), bank_missing=True
    )
    assert isinstance(inputs, ReviewedConstructionInputs)
    assert inputs.bank is None


def test_missing_review_returns_review_pending_state(tmp_path, monkeypatch):
    reviewed = _capabilities(("types",))
    _raw, _state, _entry, _bank_sha, inputs = _execute_callback(tmp_path, monkeypatch, reviewed, review_missing=True)
    assert inputs is None
