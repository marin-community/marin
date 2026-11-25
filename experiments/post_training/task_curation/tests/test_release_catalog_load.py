# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained catalog-load stage (R40 gate slice 2)."""

import dataclasses
import hashlib
import json

import pytest

from experiments.post_training.task_curation.release_stages import catalog_load as cl
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

PRIOR = "b" * 32
TARGET = "c" * 32
MEASURED_SHA = "a" * 64
MEASURED_SCHEMA = "e" * 64


def _write_json(tmp_path, name, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(uri=str(path), sha256=digest)


def _inputs(tmp_path, **overrides) -> cl.CatalogLoadInputs:
    contracts = overrides.pop("contract_rows", [])
    contracts_ref = _write_json(tmp_path, "contracts.json", {"contracts": contracts})
    base = dict(
        release_id="rtest",
        prior_load_id=PRIOR,
        target_load_id=TARGET,
        snapshot_id="assembly-test-v1",
        source_revision="rev-1",
        catalog_root="/tmp/catalog",
        receipt_root=str(tmp_path / "releases/rtest"),
        publication=_write_json(tmp_path, "publication.json", {"backend": {}, "identity": {}}),
        staging={
            "prior_qc_completion": _write_json(tmp_path, "browse.json", {}),
            "fresh_qc_index": _write_json(tmp_path, "qc.json", {}),
            "contracts": contracts_ref,
            "history_receipt": _write_json(tmp_path, "history.json", {}),
        },
        expected_source_count=2,
        expected_contract_count=len(contracts) or 1,
        expected_contract_tasks=sum(row["input"]["measured_tasks"] for row in contracts) or 10,
        expected_selected_panels=1,
        expected_browse_source_count=1,
        measured_module="does-not-matter",
        measured_revision="rev",
        measured_sha256=MEASURED_SHA,
        measured_schema_sha256=MEASURED_SCHEMA,
    )
    return cl.CatalogLoadInputs(**{**base, **overrides})


def test_inputs_validate_load_ids_and_staging(tmp_path):
    inputs = _inputs(tmp_path)
    inputs.validate()
    with pytest.raises(ValueError, match="32-hex"):
        dataclasses.replace(inputs, target_load_id="not-a-load-id").validate()
    staging = dict(inputs.staging)
    del staging["history_receipt"]
    with pytest.raises(ValueError, match="history_receipt"):
        dataclasses.replace(inputs, staging=staging).validate()


def test_read_ref_refuses_digest_mismatch(tmp_path):
    ref = _write_json(tmp_path, "doc.json", {"a": 1})
    tampered = ArtifactRef(uri=ref.uri, sha256="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        cl.read_ref(tampered)


def test_dry_run_refuses_contract_count_mismatch(tmp_path):
    inputs = _inputs(tmp_path, contract_rows=[{"input": {"measured_tasks": 10}}], expected_contract_count=2)
    with pytest.raises(ValueError, match="derived contracts"):
        cl.run_dry_run(inputs)


def test_dry_run_refuses_measured_dependency_drift(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, contract_rows=[{"input": {"measured_tasks": 10}}])
    monkeypatch.setattr(cl, "backend_schema_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="measured"):
        cl.run_dry_run(inputs)


def _aligned_state(tmp_path, monkeypatch):
    """The full aligned-state fixture: publication, catalog, measured module, review stub."""
    inputs = _inputs(tmp_path, contract_rows=[{"input": {"measured_tasks": 10}}])
    monkeypatch.setattr(cl, "backend_schema_sha256", lambda: MEASURED_SCHEMA)

    source_file = tmp_path / "task_models.py"
    source_file.write_text("# measured module\n")
    inputs = dataclasses.replace(
        inputs,
        measured_module=str(source_file),
        measured_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    )

    class FakeSnapshot:
        snapshot_id = "assembly-test-v1"

    class FakeCatalog:
        def __init__(self, root):
            self.root = root

        def current(self):
            return FakeSnapshot()

        def sources(self, snapshot_id, *, pending_stage_of=None):
            assert pending_stage_of is not None  # pre-stage Pending rows need the computed stage
            return [{"id": "s1"}, {"id": "s2"}]

    monkeypatch.setattr(cl, "CurationCatalog", FakeCatalog)
    monkeypatch.setattr(
        cl,
        "canonical_snapshot_source",
        lambda row: type(
            "S",
            (),
            {
                "id": row["id"],
                "declaration": type("D", (), {"model_dump": lambda self, mode=None: {}})(),
            },
        )(),
    )
    reviewed = {}
    monkeypatch.setattr(
        cl,
        "review_admissions",
        lambda declarations, survey, available: reviewed.update(
            {source_id: sorted(available) for source_id in declarations}
        )
        or None,
    )
    # The survey model itself is exercised in test_release_admission_check; this
    # test pins the load-level wiring.
    monkeypatch.setattr(cl, "canonical_backend_survey", lambda document: object())

    backend_path = tmp_path / "backend.json"
    backend_path.write_text(json.dumps({"sources": [{"source_id": "s1"}, {"source_id": "s2"}]}))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"fingerprints": {"s1": {}, "s2": {}}}))
    publication = {
        "backend": {"uri": str(backend_path), "sha256": hashlib.sha256(backend_path.read_bytes()).hexdigest()},
        "identity": {"uri": str(identity_path), "sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest()},
    }
    publication_path = tmp_path / "publication.json"
    publication_path.write_text(json.dumps(publication))
    inputs = dataclasses.replace(
        inputs,
        publication=ArtifactRef(
            uri=str(publication_path), sha256=hashlib.sha256(publication_path.read_bytes()).hexdigest()
        ),
    )
    return inputs, reviewed


def test_dry_run_passes_with_aligned_state(tmp_path, monkeypatch):
    inputs, reviewed = _aligned_state(tmp_path, monkeypatch)
    proof = cl.run_dry_run(inputs)
    assert proof["sql_writes"] == 0
    assert proof["target_load_id"] == TARGET
    assert proof["contracts"] == 1
    # The dry run sweeps every source through the admission review (coordinator
    # design input): the real sweep is behavior-tested in test_release_admission_check.
    assert sorted(reviewed) == ["s1", "s2"]


def test_dry_run_refuses_a_blocked_admission(tmp_path, monkeypatch):
    """A blocked source fails the dry run — never the cluster job (2026-11-23)."""
    inputs, _ = _aligned_state(tmp_path, monkeypatch)

    def blocked_review(declarations, survey, available):
        raise ValueError("Admission blocked for 1 source(s): s2: no surveyed production backend can run the source")

    monkeypatch.setattr(cl, "review_admissions", blocked_review)
    with pytest.raises(ValueError, match="Admission blocked"):
        cl.run_dry_run(inputs)


def test_verify_history_receipt_refuses_wrong_load(tmp_path):
    history = {
        "load_id": "d" * 32,
        "prior_load_id": PRIOR,
        "selected_pointer_changed": False,
        "prior_assessment_headers": 5,
    }
    inputs = _inputs(tmp_path, staging={"history_receipt": _write_json(tmp_path, "h.json", history)})
    with pytest.raises(ValueError, match="history receipt targets"):
        cl.verify_history_receipt(inputs)


def test_history_receipt_refuses_changed_pointer(tmp_path):
    history = {
        "load_id": TARGET,
        "prior_load_id": PRIOR,
        "selected_pointer_changed": True,
        "prior_assessment_headers": 5,
    }
    inputs = _inputs(tmp_path, staging={"history_receipt": _write_json(tmp_path, "h.json", history)})
    with pytest.raises(ValueError, match="changed selection pointer"):
        cl.verify_history_receipt(inputs)


def test_write_receipt_refuses_overwrite(tmp_path):
    inputs = _inputs(tmp_path)
    receipt_dir = tmp_path / "releases/rtest/catalog-load"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "receipt.json").write_text("{}\n")
    with pytest.raises(ValueError, match="already exists"):
        cl._write_receipt(inputs, {"kind": "catalog-load"})


def test_failed_run_emits_alert(tmp_path):
    feed = tmp_path / "alerts.jsonl"
    inputs = _inputs(tmp_path, staging={})
    with pytest.raises(ValueError, match="staging refs must include"):
        cl.run_catalog_load(inputs, dry_run=True, feeds=(feed,))
    rows = [json.loads(line) for line in feed.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["STAGE_START", "STAGE_FAIL"]
