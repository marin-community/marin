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

PANEL_SHA = "b" * 64
PANEL_IDENTITY = "5" * 64


def _seal_panel(tmp_path: Path) -> ArtifactRef:
    """One staged 100-record panel document, gzip-compressed, as the prior index holds it."""
    panel = {
        "items": [{"task_id": f"t{i}"} for i in range(100)],
        "raw": [{"task_id": f"t{i}"} for i in range(100)],
        "receipt": {"current_pin": "d" * 64},
    }
    raw = gzip.compress(json.dumps(panel).encode())
    path = StoragePath(str(tmp_path / "prior" / f"{PANEL_IDENTITY}.json.gz"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _prior_index(tmp_path: Path) -> ArtifactRef:
    reference = _seal_panel(tmp_path)
    document = {
        "kind": "r39-fresh-qc-index",
        "basis": "The seven prior admissions' current-pin 100-record SAMPLE QC panels.",
        "load_id": "c" * 32,
        "receipts": [
            {
                "source_id": "Marin:fixture",
                "status": "complete",
                "records": 100,
                "bytes": 1234,
                "identity": PANEL_IDENTITY,
                "reference": {"uri": reference.uri, "sha256": reference.sha256},
            }
        ],
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "prior" / "fresh-qc-index.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _completion(tmp_path: Path) -> ArtifactRef:
    document = {
        "kind": "atlas-testrelease-prior-qc-browse",
        "status": "PASS",
        "results": [{"source_id": "Marin:fixture", "original_receipt": "r" * 64}],
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "browse" / "completion.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _inputs(tmp_path: Path) -> stage.FreshQcIndexInputs:
    return stage.FreshQcIndexInputs(
        release_id="testrelease",
        target_load_id="a" * 32,
        snapshot_id="assembly-test-v1",
        source_revision="e" * 40,
        output_root=str(tmp_path / "release"),
        prior_index=_prior_index(tmp_path),
        qc_completion=_completion(tmp_path),
    )


def test_write_restages_the_panel_byte_identically_and_binds_the_release(tmp_path):
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=False)
    assert proof["load_id"] == "a" * 32 and proof["snapshot_id"] == "assembly-test-v1"
    index = json.loads(StoragePath(proof["index"]["uri"]).read_text())
    row = index["receipts"][0]
    assert row["source_id"] == "Marin:fixture" and row["status"] == "complete" and row["records"] == 100
    # the staged bytes are the prior panel's bytes, digest-verified
    staged = StoragePath(row["reference"]["uri"])
    prior = json.loads(StoragePath(_prior_index(tmp_path).uri).read_text())["receipts"][0]["reference"]
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == prior["sha256"]
    assert index["provenance"]["prior_index"]["sha256"] == inputs.prior_index.sha256


def test_dry_run_verifies_without_writing(tmp_path):
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=True)
    assert proof["panels"] == 1
    assert not StoragePath(f"{inputs.output_root}/{stage.STAGED_DIR}").exists()
    assert not StoragePath(f"{inputs.output_root}/fresh-qc-index.json").exists()


def test_refuses_a_panel_the_browse_never_staged(tmp_path):
    inputs = _inputs(tmp_path)
    completion = StoragePath(inputs.qc_completion.uri)
    document = json.loads(completion.read_text())
    document["results"] = []
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    completion.write_bytes(raw, auto_mkdir=True)
    broken = dataclasses.replace(
        inputs, qc_completion=ArtifactRef(uri=str(completion), sha256=hashlib.sha256(raw).hexdigest())
    )
    with pytest.raises(ValueError, match="no staged panels"):
        stage.run_fresh_qc_index(broken, dry_run=True)


def test_refuses_an_incomplete_prior_panel(tmp_path):
    inputs = _inputs(tmp_path)
    index = StoragePath(inputs.prior_index.uri)
    document = json.loads(index.read_text())
    document["receipts"][0]["records"] = 99
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    index.write_bytes(raw, auto_mkdir=True)
    broken = dataclasses.replace(inputs, prior_index=ArtifactRef(uri=str(index), sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match="99 records"):
        stage.run_fresh_qc_index(broken, dry_run=True)
