# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained curriculum-load stage (R40 gate slice 2)."""

import dataclasses
import hashlib
import json

import pytest

from experiments.post_training.task_curation.release_stages import curriculum_load as cul
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

PRIOR_CURRICULUM = "1" * 32
CATALOG_PRIOR = "2" * 32
CATALOG_TARGET = "3" * 32


def _write_json(tmp_path, name, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _handoff_document(tmp_path, sample_task_count=7) -> dict:
    coverage = _write_json(tmp_path, "coverage.json", {"coverage_id": "cov-1", "sample_task_count": sample_task_count})
    release = _write_json(tmp_path, "release.json", {"release_id": "cov-1"})
    confidence = _write_json(tmp_path, "confidence.json", {"rows": []})
    return {
        "source_count": 2,
        "inventory_count": 3,
        "task_count": 900,
        "all_retained_assignment_values_verified": True,
        "all_statistics_and_confidence_values_unchanged": True,
        "all_task_ids_unchanged": True,
        "all_public_inputs_unchanged": True,
        "rebound_source": "Test:Rebound",
        "rebound_tasks": 5,
        "confidence_validated_rows": 11,
        "release": {"uri": release.uri, "sha256": release.sha256},
        "coverage": {"uri": coverage.uri, "sha256": coverage.sha256},
        "confidence": {"uri": confidence.uri, "sha256": confidence.sha256},
    }


def _inputs(tmp_path, **overrides) -> cul.CurriculumLoadInputs:
    handoff = overrides.pop("handoff_document", None) or _handoff_document(tmp_path)
    scan = overrides.pop("scan_document", None)
    if scan is None:
        scan = {
            "status": "PASS",
            "handoff": None,
            "codec": cul.CATALOG_TEXT_CODEC,
            "artifacts": {
                "release": handoff["release"],
                "coverage": handoff["coverage"],
                "confidence": handoff["confidence"],
            },
            "task_count": handoff["task_count"],
            "sample_task_count": 7,
            "source_count": handoff["source_count"],
        }
    handoff_ref = _write_json(tmp_path, "handoff.json", handoff)
    scan["handoff"] = handoff_ref.model_dump(mode="json")
    scan_ref = _write_json(tmp_path, "text-scan.json", scan)
    base = dict(
        release_id="rtest",
        prior_curriculum_load_id=PRIOR_CURRICULUM,
        catalog_prior_load_id=CATALOG_PRIOR,
        catalog_target_load_id=CATALOG_TARGET,
        source_revision="rev-1",
        receipt_root=str(tmp_path / "releases/rtest"),
        handoff=handoff_ref,
        text_scan=scan_ref,
        expected_source_count=3,
        expected_inventory_count=3,
        expected_aligned_sources=2,
        expected_task_count=900,
        expected_rebound_source="Test:Rebound",
        expected_rebound_tasks=5,
        expected_confidence_rows=11,
    )
    return cul.CurriculumLoadInputs(**{**base, **overrides})


def test_inputs_validate_load_ids(tmp_path):
    _inputs(tmp_path).validate()
    with pytest.raises(ValueError, match="32-hex"):
        dataclasses.replace(_inputs(tmp_path), catalog_target_load_id="short").validate()


def test_verify_handoff_refuses_changed_values(tmp_path):
    handoff = _handoff_document(tmp_path)
    handoff["all_task_ids_unchanged"] = False
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="changed composition values"):
        cul.verify_handoff(inputs, handoff)


def test_verify_handoff_refuses_count_drift(tmp_path):
    handoff = _handoff_document(tmp_path)
    handoff["task_count"] = 901
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="tasks"):
        cul.verify_handoff(inputs, handoff)


def test_verify_text_scan_refuses_wrong_handoff(tmp_path):
    handoff = _handoff_document(tmp_path)
    other = _write_json(tmp_path, "other.json", handoff)
    scan = {
        "status": "PASS",
        "handoff": other.model_dump(mode="json"),
        "codec": cul.CATALOG_TEXT_CODEC,
        "artifacts": {
            "release": handoff["release"],
            "coverage": handoff["coverage"],
            "confidence": handoff["confidence"],
        },
        "task_count": handoff["task_count"],
        "sample_task_count": 7,
        "source_count": handoff["source_count"],
    }
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="different handoff"):
        cul.verify_text_scan(handoff, inputs.handoff, scan, {"sample_task_count": 7})


def test_verify_text_scan_refuses_codec_drift(tmp_path):
    inputs = _inputs(tmp_path, scan_document=None)
    document = json.loads((tmp_path / "text-scan.json").read_text())
    document["codec"] = "other-codec"
    (tmp_path / "text-scan.json").write_text(json.dumps(document))
    inputs = dataclasses.replace(
        inputs,
        text_scan=cul.ArtifactRef(
            uri=str(tmp_path / "text-scan.json"),
            sha256=hashlib.sha256((tmp_path / "text-scan.json").read_bytes()).hexdigest(),
        ),
    )
    handoff = json.loads((tmp_path / "handoff.json").read_text())
    scan = json.loads((tmp_path / "text-scan.json").read_text())
    with pytest.raises(ValueError, match="loader codec"):
        cul.verify_text_scan(handoff, inputs.handoff, scan, {"sample_task_count": 7})


def test_dry_run_validates_with_zero_sql(tmp_path):
    inputs = _inputs(tmp_path)
    proof = cul.run_curriculum_load(inputs, dry_run=True, feeds=(tmp_path / "alerts.jsonl",))
    assert proof["sql_writes"] == 0
    assert proof["prior_curriculum_load_id"] == PRIOR_CURRICULUM
    assert proof["tasks"] == 900


def test_failed_run_emits_alert(tmp_path):
    feed = tmp_path / "alerts.jsonl"
    handoff = _handoff_document(tmp_path)
    handoff["source_count"] = 99
    inputs = _inputs(tmp_path, handoff_document=handoff, expected_source_count=2)
    with pytest.raises(ValueError, match="sources"):
        cul.run_curriculum_load(inputs, dry_run=True, feeds=(feed,))
    rows = [json.loads(line) for line in feed.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["STAGE_START", "STAGE_FAIL"]


def test_verify_handoff_separates_aligned_and_catalog_sources(tmp_path):
    """Compose aligns only measured sources (R40: 305 of 321); the inventory carries all of them."""
    handoff = _handoff_document(tmp_path)
    cul.verify_handoff(_inputs(tmp_path), handoff)
    with pytest.raises(ValueError, match="handoff aligns 2 sources, expected 3"):
        cul.verify_handoff(_inputs(tmp_path, expected_aligned_sources=3), handoff)
    with pytest.raises(ValueError, match="handoff inventories 3 sources"):
        cul.verify_handoff(_inputs(tmp_path, expected_source_count=4), handoff)


def test_resume_cursor_starts_fresh_while_the_prior_curriculum_is_selected():
    """R40: the stage passed the selected prior load as a resume cursor and the loader refused."""
    assert cul.resume_cursor([], PRIOR_CURRICULUM) is None
    assert cul.resume_cursor([{"load_id": PRIOR_CURRICULUM, "previous_load_id": "e" * 32}], PRIOR_CURRICULUM) is None
    earlier_attempt = {"load_id": "f" * 32, "previous_load_id": PRIOR_CURRICULUM}
    assert cul.resume_cursor([earlier_attempt], PRIOR_CURRICULUM) == "f" * 32
    with pytest.raises(ValueError, match="unrelated load"):
        cul.resume_cursor([{"load_id": "f" * 32, "previous_load_id": "e" * 32}], PRIOR_CURRICULUM)
