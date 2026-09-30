import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from test_pipeline import _portfolio_review, _proposed_plan, _valid_proposal

from capability_pipeline import cli
from capability_pipeline import proposal_adoption as adoption
from capability_pipeline.inference import digest
from capability_pipeline.validation import PARTIAL_ADMISSION_POLICY


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bind_snapshot(root: Path):
    files = {
        p.relative_to(root).as_posix(): _sha(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name not in {"snapshot-capture.json", "pull-manifest.json"}
    }
    remote = {"snapshot_id": "snapshot-1", "created_utc": "2026-09-21T00:00:00Z",
              "final": False, "files": files, "omitted": {}}
    raw = json.dumps(remote, sort_keys=True)
    remote_sha = _sha(raw.encode())
    _write(root / "snapshot-capture.json", {
        "source": "s3://example/frozen", "prefixes": [],
        "remote_manifest_json": raw, "remote_manifest_sha256": remote_sha,
    })
    _write(root / "pull-manifest.json", {
        "source": "s3://example/frozen", "remote_snapshot_id": "snapshot-1",
        "remote_manifest_sha256": remote_sha, "complete_manifest": True,
        "complete_snapshot": True, "omitted": {}, "files": files,
    })


def _fixture(tmp_path, monkeypatch, cap="cap.one"):
    capability = {"capability_id": cap, "subject_id": "subject.one",
                  "capability": {"id": cap}}
    pilot = {"source": {"sha256": "a" * 64}, "capabilities": [capability]}
    monkeypatch.setattr(adoption, "load_pilot", lambda _: pilot)
    pilot_path = tmp_path / "pilot.json"
    _write(pilot_path, pilot)
    source = tmp_path / "snapshot"
    proposal = source / "proposal"
    proposal.mkdir(parents=True)
    plan = _proposed_plan(cap)
    props = [_valid_proposal(cap, slot) for slot in range(1, 11)]
    review = _portfolio_review(cap, range(1, 11), "accept", [])
    accepted = [{"proposal": prop, "review": next(r for r in review["reviews"] if r["slot"] == prop["slot"]),
                 "proposal_hash": digest(prop), "provenance": cli._source_provenance(pilot, capability)}
                for prop in props]
    documents = {
        "accepted.json": accepted, "rejected.json": [], "null.json": [],
        "plans.json": {cap: plan}, "proposals.json": props,
        "reviews-round-1.json": {cap: review}, "input_pilot.json": pilot,
        "run.json": {"stage": "propose", "capability_ids": [cap],
                     "pilot_hash": digest(pilot), "repair_rounds": 1},
        "report.json": {"stage": "proposal_review", "state": "complete",
                        "capabilities": 1, "expected_slots": 10, "accepted": 10,
                        "rejected_or_needs_repair": 0, "null": 0,
                        "missing_slots": [], "environment_counts": {
                            props[0]["environment"]: 10},
                        "verifier_counts": {props[0]["verification"]: 10},
                        "failures": {"proposal": {}, "repair": {}}},
    }
    for name, document in documents.items():
        _write(proposal / name, document)
    source_file = b"# frozen controller\n"
    source_contents = {
        "capability_pipeline/test.py": source_file,
        "inputs/pilot.json": pilot_path.read_bytes(),
    }
    package_hash = digest({"test.py": _sha(source_file)})
    original_run = {
        "schema_version": "capability-generate-v1", "pilot_sha256": _sha(pilot_path.read_bytes()),
        "settings": {"controller_package_sha256": package_hash,
                     "proposal_repair_rounds": 1, "construction_max_repair_rounds": 2},
    }
    original_run["identity_sha256"] = digest(original_run)
    _write(source / "generate-run.json", original_run)
    _bind_snapshot(source)
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        manifest = {"files": {name: _sha(data) for name, data in source_contents.items()}}
        contents = {**source_contents, "manifest.json": json.dumps(manifest).encode()}
        for name, data in contents.items():
            header = tarfile.TarInfo(name)
            header.size = len(data)
            archive.addfile(header, io.BytesIO(data))
    launch_path = tmp_path / "launch.json"
    _write(launch_path, {
        "source_snapshot_sha256": adoption._sha(archive_path),
        "source_snapshot_files": 2,
        "controller_files": {"capability_pipeline/test.py": _sha(source_file)},
        "manifest_sha256": _sha(pilot_path.read_bytes()),
        "manifest_capabilities": 1, "proposal_slots": 10,
    })
    return source, archive_path, launch_path, pilot_path


def test_read_only_checkpoint_preflight_and_adoption(tmp_path, monkeypatch):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    before = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    checked = adoption.validate_checkpoint(source, archive, launch, pilot)
    assert checked["slot_accounting"] == {
        "expected_slots": 10, "accepted": 10, "rejected": 0, "null": 0, "missing": 0,
    }
    assert before == sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    target = tmp_path / "target"
    target.mkdir()
    target_identity = {"schema_version": "capability-generate-v1",
                       "pilot_sha256": adoption._sha(pilot),
                       "settings": {"proposal_repair_rounds": 1,
                                    "construction_max_repair_rounds": 2}}
    target_identity["identity_sha256"] = digest(target_identity)
    _write(target / "generate-run.json", target_identity)
    (target / "input-pilot.json").write_bytes(pilot.read_bytes())
    receipt = adoption.validate_and_adopt(source, archive, launch, pilot, target, target_identity)
    assert receipt["accepted_count"] == 10
    assert receipt["adoption"]["inference_replayed"] is False
    assert receipt["adoption"]["acceptance_reconstructed"] is True
    assert (target / "proposal" / "accepted.json").read_bytes() == (source / "proposal" / "accepted.json").read_bytes()
    with pytest.raises(adoption.AdoptionError, match="already contains"):
        adoption.validate_and_adopt(source, archive, launch, pilot, target, target_identity)


def test_checkpoint_reconstructs_partial_portfolio_admission(tmp_path, monkeypatch):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    root = source / "proposal"
    reviews = json.loads((root / "reviews-round-1.json").read_text())
    review = reviews["cap.one"]
    review["portfolio_verdict"] = "repair"
    review["portfolio_issues"] = ["Slot 1 requires a correction."]
    review["reviews"][0].update(
        verdict="repair", required_changes=["Correct the anchor."]
    )
    _write(root / "reviews-round-1.json", reviews)
    accepted = json.loads((root / "accepted.json").read_text())
    first = accepted.pop(0)
    _write(root / "accepted.json", accepted)
    _write(root / "rejected.json", [{
        "proposal": first["proposal"], "review": review["reviews"][0],
        "portfolio_verdict": "repair",
    }])
    report = json.loads((root / "report.json").read_text())
    report.update(
        state="needs_iteration", accepted=9, rejected_or_needs_repair=1,
        acceptance_policy=PARTIAL_ADMISSION_POLICY,
        partial_portfolio_admissions=9, invalid_final_plans={},
        environment_counts={first["proposal"]["environment"]: 9},
        verifier_counts={first["proposal"]["verification"]: 9},
    )
    _write(root / "report.json", report)
    _bind_snapshot(source)
    checked = adoption.validate_checkpoint(source, archive, launch, pilot)
    assert checked["slot_accounting"] == {
        "expected_slots": 10, "accepted": 9, "rejected": 1, "null": 0,
        "missing": 0,
    }


def test_checkpoint_rejects_unlisted_file_and_forged_acceptance(tmp_path, monkeypatch):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    (source / "proposal" / "unlisted.txt").write_text("unverified")
    with pytest.raises(adoption.AdoptionError, match="undeclared files"):
        adoption.validate_checkpoint(source, archive, launch, pilot)
    (source / "proposal" / "unlisted.txt").unlink()
    accepted = json.loads((source / "proposal" / "accepted.json").read_text())
    accepted[0]["proposal_hash"] = "0" * 64
    _write(source / "proposal" / "accepted.json", accepted)
    _bind_snapshot(source)
    with pytest.raises(adoption.AdoptionError, match="accepted inventory differs"):
        adoption.validate_checkpoint(source, archive, launch, pilot)


def test_checkpoint_rejects_construction_and_archive_drift(tmp_path, monkeypatch):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    archive.write_bytes(archive.read_bytes() + b"drift")
    with pytest.raises(adoption.AdoptionError, match="source archive differs"):
        adoption.validate_checkpoint(source, archive, launch, pilot)
    source, archive, launch, pilot = _fixture(tmp_path / "fresh", monkeypatch)
    _write(source / "construction-progress.json", {"state": "running"})
    _bind_snapshot(source)
    with pytest.raises(adoption.AdoptionError, match="already contains construction"):
        adoption.validate_checkpoint(source, archive, launch, pilot)


@pytest.mark.parametrize("mutation", ["review_verdict", "review_score", "missing_slot", "provenance"])
def test_rehashed_checkpoint_still_rejects_inconsistent_semantics(tmp_path, monkeypatch, mutation):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    proposal = source / "proposal"
    if mutation.startswith("review"):
        path = proposal / "reviews-round-1.json"
        value = json.loads(path.read_text())
        row = value["cap.one"]["reviews"][0]
        if mutation == "review_verdict":
            row["verdict"] = "repair"
        else:
            row["scores"][next(iter(row["scores"]))] = 1
    elif mutation == "missing_slot":
        path = proposal / "report.json"
        value = json.loads(path.read_text())
        value["missing_slots"] = ["cap.one:1"]
    else:
        path = proposal / "accepted.json"
        value = json.loads(path.read_text())
        value[0]["provenance"]["capability_record_hash"] = "0" * 64
    _write(path, value)
    _bind_snapshot(source)
    with pytest.raises((adoption.AdoptionError, ValueError)):
        adoption.validate_checkpoint(source, archive, launch, pilot)


def test_adoption_rejects_budget_and_pilot_drift(tmp_path, monkeypatch):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    target = tmp_path / "target"
    target.mkdir()
    (target / "input-pilot.json").write_bytes(pilot.read_bytes())
    identity = {"schema_version": "capability-generate-v1", "pilot_sha256": adoption._sha(pilot),
                "settings": {"proposal_repair_rounds": 1, "construction_max_repair_rounds": 3}}
    identity["identity_sha256"] = digest(identity)
    _write(target / "generate-run.json", identity)
    with pytest.raises(adoption.AdoptionError, match="repair budgets"):
        adoption.validate_and_adopt(source, archive, launch, pilot, target, identity)
    identity["pilot_sha256"] = "0" * 64
    identity["identity_sha256"] = digest({k: v for k, v in identity.items() if k != "identity_sha256"})
    _write(target / "generate-run.json", identity)
    with pytest.raises(adoption.AdoptionError, match="target pilot differs"):
        adoption.validate_and_adopt(source, archive, launch, pilot, target, identity)


def test_held_capability_cannot_be_adopted_as_accepted(tmp_path, monkeypatch):
    source, archive, launch, pilot = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(adoption, "_HELD", frozenset({"cap.one"}))
    with pytest.raises(adoption.AdoptionError, match="held capability"):
        adoption.validate_checkpoint(source, archive, launch, pilot)
