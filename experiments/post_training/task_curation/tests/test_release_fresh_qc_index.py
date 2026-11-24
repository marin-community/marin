# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the fresh-QC-index stage (the graph's staged-panel index producer)."""

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import fresh_qc_index as stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

REVIEW_SHA = "b" * 64


def _review_artifact(tmp_path: Path) -> dict:
    return {"uri": f"file://{tmp_path}/review/.artifact.json", "sha256": REVIEW_SHA}


def _declarations(tmp_path: Path) -> ArtifactRef:
    document = {
        "Marin:fixture": {
            "id": "Marin:fixture",
            "data_restriction": "open",
            "independent_qc": {"evidence_url": f"file://{tmp_path}/review"},
        }
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "declarations.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _completion(tmp_path: Path, *, status: str = "PASS") -> ArtifactRef:
    document = {
        "kind": "atlas-testrelease-prior-qc-browse",
        "status": status,
        "results": [
            {
                "source_id": "Marin:fixture",
                "original_receipt": {"artifact": _review_artifact(tmp_path), "current_pin": True},
            }
        ],
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "browse" / "completion.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _inputs(tmp_path: Path, **overrides) -> stage.FreshQcIndexInputs:
    inputs = stage.FreshQcIndexInputs(
        release_id="testrelease",
        target_load_id="a" * 32,
        snapshot_id="assembly-test-v1",
        source_revision="e" * 40,
        output_root=str(tmp_path / "release"),
        declarations=_declarations(tmp_path),
        qc_completion=_completion(tmp_path),
    )
    return dataclasses.replace(inputs, **overrides) if overrides else inputs


@pytest.fixture
def staged_panel(monkeypatch, tmp_path: Path):
    """The maintained stager returns a verified staged document without touching storage."""
    captured = {}

    def fake_stage(reference, source, workers, mapping_reference):
        captured["reference"] = reference
        captured["source"] = source
        return {
            "data_restriction": "open",
            "assessment": {"source_id": source["id"]},
            "populations": [],
            "items": [{"task_id": f"t{i}"} for i in range(100)],
            "raw": [],
            "batches": [],
            "batch_links": [],
            "receipt": {
                "artifact": reference.model_dump(mode="json"),
                "panel": {},
                "current_pin": True,
                "mapping": None,
            },
        }

    monkeypatch.setattr("infra.marina.apps.rl_data_catalog.load_reviews.stage_content_reviews", fake_stage)
    return captured


def test_write_stages_each_panel_and_binds_the_release(tmp_path, staged_panel):
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=False)
    assert proof["load_id"] == "a" * 32 and proof["snapshot_id"] == "assembly-test-v1"
    index = json.loads(StoragePath(proof["index"]["uri"]).read_text())
    row = index["receipts"][0]
    assert row["source_id"] == "Marin:fixture" and row["status"] == "complete" and row["records"] == 100
    assert row["reference"]["uri"].endswith(f"staged-v1/{REVIEW_SHA}.json.gz")
    document = json.loads(gzip.decompress(StoragePath(row["reference"]["uri"]).read_bytes()))
    assert document["receipt"]["artifact"]["sha256"] == REVIEW_SHA
    assert index["provenance"]["declarations"]["sha256"] == inputs.declarations.sha256


def test_dry_run_verifies_without_writing(tmp_path, staged_panel):
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=True)
    assert proof["panels"] == 1
    assert not StoragePath(f"{inputs.output_root}/{stage.STAGED_DIR}").exists()
    assert not StoragePath(f"{inputs.output_root}/fresh-qc-index.json").exists()


def test_refuses_a_panel_whose_staged_receipt_differs_from_the_browse_copy(tmp_path, monkeypatch):
    def fake_stage(reference, source, workers, mapping_reference):
        return {
            "data_restriction": "open",
            "assessment": {"source_id": source["id"]},
            "populations": [],
            "items": [],
            "raw": [],
            "batches": [],
            "batch_links": [],
            "receipt": {"artifact": {"uri": "file://other", "sha256": "f" * 64}, "panel": {}, "current_pin": True},
        }

    monkeypatch.setattr("infra.marina.apps.rl_data_catalog.load_reviews.stage_content_reviews", fake_stage)
    with pytest.raises(ValueError, match="differs from the browse copy"):
        stage.run_fresh_qc_index(_inputs(tmp_path), dry_run=True)


def test_refuses_a_panel_without_a_frozen_declaration(tmp_path, staged_panel):
    inputs = _inputs(tmp_path)
    declarations = StoragePath(inputs.declarations.uri)
    document = json.loads(declarations.read_text())
    document["Marin:other"] = {"id": "Marin:other", "data_restriction": "open"}
    document.pop("Marin:fixture")
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    declarations.write_bytes(raw, auto_mkdir=True)
    broken = dataclasses.replace(
        inputs, declarations=ArtifactRef(uri=str(declarations), sha256=hashlib.sha256(raw).hexdigest())
    )
    with pytest.raises(ValueError, match="no frozen declaration"):
        stage.run_fresh_qc_index(broken, dry_run=True)


def test_refuses_a_failed_browse_completion(tmp_path, staged_panel):
    inputs = _inputs(tmp_path)
    path = StoragePath(inputs.qc_completion.uri)
    document = json.loads(path.read_text())
    document["status"] = "FAIL"
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw, auto_mkdir=True)
    broken = dataclasses.replace(
        inputs, qc_completion=ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())
    )
    with pytest.raises(ValueError, match="needs a PASS"):
        stage.run_fresh_qc_index(broken, dry_run=True)
