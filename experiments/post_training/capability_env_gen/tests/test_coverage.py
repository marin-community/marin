import json
from pathlib import Path

import pytest

from capability_pipeline.catalog import build_pilot, ingest_catalog
from capability_pipeline.coverage import CoverageError, coverage_report
from capability_pipeline.inference import digest


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _fixture(tmp_path):
    cap = {"id": "d01.one", "kind": "capability", "parent_id": None, "name": "One",
           "outcome": "Solve one.", "includes": ["one"], "excludes": ["other"],
           "prerequisites": [], "sample_tasks": []}
    catalog = {"catalog_version": "test", "curricula": [{"routing_facet": "subject_domain",
        "curriculum": {"subject_id": "D01", "subject_name": "One", "version": "test",
                       "sections": [cap]}}]}
    _write(tmp_path / "catalog.json", catalog)
    manifest = build_pilot(ingest_catalog(tmp_path / "catalog.json"), [("d01.one", "fixture")], source_path="catalog.json")
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest)
    proposal = {"capability_id": "d01.one", "slot": 1, "status": "proposed"}
    accepted = {"proposal": proposal, "proposal_hash": digest(proposal), "review": {"verdict": "accept"}}
    root = tmp_path / "proposal"
    _write(root / "input_pilot.json", manifest)
    _write(root / "run.json", {"stage": "propose", "pilot_hash": digest(manifest),
        "capability_ids": ["d01.one"]})
    _write(root / "accepted.json", [accepted])
    _write(root / "rejected.json", [])
    _write(root / "null.json", [])
    _write(root / "report.json", {"stage": "proposal_review", "state": "needs_iteration",
        "capabilities": 1, "expected_slots": 10,
        "accepted": 1, "rejected_or_needs_repair": 0, "null": 0,
        "missing_slots": [f"d01.one:{i}" for i in range(2, 11)]})
    return manifest_path, root, accepted


def test_full_denominator_and_terminal_synthesis(tmp_path):
    manifest, proposal_root, accepted = _fixture(tmp_path)
    first = coverage_report(manifest, proposal_root)
    assert first["slots_total"] == 10
    assert first["slot_dispositions"] == {"missing_proposal": 9, "pending_construction": 1}
    assert not first["accounting_complete"]
    synth = tmp_path / "synth"
    row = {"key": "d01.one:1", "proposal_hash": accepted["proposal_hash"],
           "state": "quality_accepted", "runtime_validated": True,
           "taskcompendium": {"id": "synthetic/one"}}
    _write(synth / "tasks.json", [row])
    _write(synth / "report.json", {"stage": "synthesize", "state": "needs_continuation",
        "completed_items": 1, "states": {"quality_accepted": 1}})
    _write(synth / "items" / f"one-{accepted['proposal_hash'][:12]}" / "status.json", row)
    result = coverage_report(manifest, proposal_root, [synth])
    assert result["slot_dispositions"]["quality_accepted"] == 1
    assert not result["accounting_complete"]
    assert result["capabilities_with_accepted_tasks"] == 1


def test_rejects_unexpected_duplicate_and_stale_hashes(tmp_path):
    manifest, root, accepted = _fixture(tmp_path)
    rows = _read(root / "accepted.json")
    rows[0]["proposal"]["capability_id"] = "outside"
    _write(root / "accepted.json", rows)
    with pytest.raises(CoverageError, match="out-of-manifest"):
        coverage_report(manifest, root)
    rows[0] = accepted
    _write(root / "accepted.json", rows)
    _write(root / "rejected.json", [accepted])
    with pytest.raises(CoverageError, match="duplicate"):
        coverage_report(manifest, root)
    _write(root / "rejected.json", [])
    rows[0] = {**accepted, "proposal_hash": "0" * 64}
    _write(root / "accepted.json", rows)
    with pytest.raises(CoverageError, match="mismatched"):
        coverage_report(manifest, root)


def test_missing_terminal_evidence_fails_closed(tmp_path):
    manifest, root, _ = _fixture(tmp_path)
    synth = tmp_path / "synth"
    _write(synth / "report.json", {"stage": "synthesize", "state": "complete",
        "completed_items": 1, "states": {"quality_accepted": 1}})
    with pytest.raises(CoverageError, match="missing"):
        coverage_report(manifest, root, [synth])


def test_synthesis_rejects_stale_hash_and_duplicate_identity(tmp_path):
    manifest, root, accepted = _fixture(tmp_path)
    synth = tmp_path / "synth"
    row = {"key": "d01.one:1", "proposal_hash": "0" * 64,
           "state": "pending_build", "runtime_validated": False}
    _write(synth / "tasks.json", [row])
    _write(synth / "report.json", {"stage": "synthesize", "state": "needs_continuation",
        "completed_items": 1, "states": {"pending_build": 1}})
    with pytest.raises(CoverageError, match="hash mismatch"):
        coverage_report(manifest, root, [synth])
    row["proposal_hash"] = accepted["proposal_hash"]
    _write(synth / "tasks.json", [row, row])
    _write(synth / "report.json", {"stage": "synthesize", "state": "needs_continuation",
        "completed_items": 2, "states": {"pending_build": 2}})
    _write(synth / "items" / f"one-{accepted['proposal_hash'][:12]}" / "status.json", row)
    with pytest.raises(CoverageError, match="duplicate synthesis"):
        coverage_report(manifest, root, [synth])


def test_nulls_can_complete_accounting_without_quality_yield(tmp_path):
    manifest, root, _ = _fixture(tmp_path)
    _write(root / "accepted.json", [])
    _write(root / "null.json", [
        {"capability_id": "d01.one", "slot": slot, "status": "null"}
        for slot in range(1, 11)
    ])
    _write(root / "report.json", {"stage": "proposal_review", "state": "needs_iteration",
        "capabilities": 1, "expected_slots": 10, "accepted": 0,
        "rejected_or_needs_repair": 0, "null": 10, "missing_slots": []})
    result = coverage_report(manifest, root)
    assert result["accounting_complete"]
    assert result["coverage_complete"]
    assert not result["all_slots_quality_accepted"]
    assert result["capabilities_with_accepted_tasks"] == 0


def test_failed_unknown_is_unresolved_and_status_must_agree(tmp_path):
    manifest, root, accepted = _fixture(tmp_path)
    synth = tmp_path / "synth"
    row = {"key": "d01.one:1", "proposal_hash": accepted["proposal_hash"],
           "state": "failed", "runtime_validated": False,
           "issues": ["unexpected worker transport failure"]}
    _write(synth / "tasks.json", [row])
    _write(synth / "report.json", {"stage": "synthesize", "state": "needs_continuation",
        "completed_items": 1, "states": {"failed": 1}})
    status = synth / "items" / f"one-{accepted['proposal_hash'][:12]}" / "status.json"
    _write(status, {**row, "runtime_validated": True})
    with pytest.raises(CoverageError, match="authoritative"):
        coverage_report(manifest, root, [synth])
    _write(status, row)
    result = coverage_report(manifest, root, [synth])
    assert result["slot_dispositions"]["execution_failure_unresolved"] == 1
    assert not result["accounting_complete"]


def test_missing_slot_inventory_must_be_exact(tmp_path):
    manifest, root, _ = _fixture(tmp_path)
    report = _read(root / "report.json")
    report["missing_slots"].pop()
    _write(root / "report.json", report)
    with pytest.raises(CoverageError, match="missing-slot inventory"):
        coverage_report(manifest, root)


def test_typed_exhausted_repair_marker_is_terminal_rejection(tmp_path):
    manifest, root, accepted = _fixture(tmp_path)
    synth = tmp_path / "synth"
    row = {"key": "d01.one:1", "proposal_hash": accepted["proposal_hash"],
           "state": "pending_attack_adjudication", "runtime_validated": False,
           "terminal_disposition": "rejected",
           "repair_budget": {"used": 2, "max": 2, "exhausted": True}}
    _write(synth / "tasks.json", [row])
    _write(synth / "report.json", {"stage": "synthesize", "state": "needs_continuation",
        "completed_items": 1, "states": {"pending_attack_adjudication": 1}})
    status = synth / "items" / f"one-{accepted['proposal_hash'][:12]}" / "status.json"
    _write(status, row)
    assert coverage_report(manifest, root, [synth])["slot_dispositions"]["rejected"] == 1
    row["repair_budget"]["exhausted"] = False
    _write(synth / "tasks.json", [row])
    _write(status, row)
    with pytest.raises(CoverageError, match="invalid exhausted"):
        coverage_report(manifest, root, [synth])


def _read(path):
    return json.loads(Path(path).read_text())
