from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from capability_pipeline.catalog import (
    CatalogError,
    build_pilot,
    build_subject_cohort,
    canonical_sha256,
    ingest_catalog,
    load_pilot,
    validate_pilot,
)


def test_new_catalog_progression_and_deterministic_subject_cohort(
    tmp_path: Path,
) -> None:
    a = _capability("d01.one")
    b = _capability("d01.two")
    c = _capability("d02.one")
    first = _wrapper("D01", a)
    first["curriculum"]["sections"].append(b)
    progression = {
        "catalog_version": "new-test",
        "prompt_version": "p1",
        "scope_subject_ids": ["D01", "D02"],
        "edges": [{"prerequisite_id": "d01.one", "dependent_id": "d02.one"}],
    }
    path = tmp_path / "new_catalog.json"
    path.write_text(
        json.dumps(
            {
                "catalog_version": "new-test",
                "curricula": [first, _wrapper("D02", c)],
                "learning_progression": progression,
            }
        )
    )
    ingestion = ingest_catalog(path)
    cohort = build_subject_cohort(ingestion, source_path=str(path))
    validate_pilot(cohort)
    assert len(cohort["capabilities"]) == 2
    assert {row["subject_id"] for row in cohort["capabilities"]} == {"D01", "D02"}
    assert cohort == build_subject_cohort(ingestion, source_path=str(path))
    context = cohort["learning_progression"]
    assert context["source_sha256"] == canonical_sha256(progression)
    assert context["edges"] == progression["edges"]
    context["edges"][0]["dependent_id"] = "tampered"
    with pytest.raises(CatalogError, match="edge digest"):
        validate_pilot(cohort)


ROOT = Path(__file__).resolve().parents[1]


def _capability(capability_id: str, name: str = "Capability") -> dict[str, object]:
    return {
        "id": capability_id,
        "kind": "capability",
        "parent_id": None,
        "name": name,
        "outcome": f"Do {name.lower()} with evidence.",
        "includes": ["complete behavior"],
        "excludes": ["unrelated behavior"],
        "prerequisites": [],
        "sample_tasks": [{"kind": "entry", "instruction": "Produce an artifact."}],
    }


def _wrapper(subject_id: str, capability: dict[str, object]) -> dict[str, object]:
    return {
        "routing_facet": "subject_domain",
        "curriculum": {
            "version": "test-1",
            "subject_id": subject_id,
            "subject_name": f"Subject {subject_id}",
            "sections": [capability],
        },
    }


def test_ingest_complete_catalog(tmp_path: Path) -> None:
    capability = _capability("subject.one")
    source = {
        "catalog_version": "test",
        "curricula": [_wrapper("S01", capability)],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    ingestion = ingest_catalog(path)

    assert ingestion.catalog_version == "test"
    assert ingestion.audit.status == "complete"
    assert ingestion.audit.dropped_bytes == 0
    assert ingestion.audit.dropped_records_exact == 0
    assert ingestion.audit.complete_curricula == 1
    assert [record.capability_id for record in ingestion.records] == ["subject.one"]
    assert ingestion.records[0].capability == capability
    assert ingestion.records[0].subject_id == "S01"


def test_salvage_emits_only_complete_objects_and_audits_loss(tmp_path: Path) -> None:
    first_wrapper = json.dumps(
        _wrapper("S01", _capability("subject.one")), separators=(",", ":")
    )
    complete_orphan = json.dumps(_capability("subject.two"), separators=(",", ":"))
    # The final capability exposes an id, but ends in a partial JSON string. It must
    # never be returned as a record.
    damaged_tail = '{"id":"subject.three","kind":"capability","name":"part'
    data = (
        '{"catalog_version":"test","curricula":['
        + first_wrapper
        + ',{"routing_facet":"subject_domain","curriculum":{"sections":['
        + complete_orphan
        + ","
        + damaged_tail
    ).encode("utf-8") + (b"\0" * 17)
    path = tmp_path / "catalog.json"
    path.write_bytes(data)

    ingestion = ingest_catalog(path)

    assert ingestion.audit.status == "incomplete"
    assert ingestion.audit.complete_curricula == 1
    assert ingestion.audit.incomplete_curricula == 1
    assert ingestion.audit.dropped_records_minimum == 1
    assert ingestion.audit.dropped_records_exact is None
    assert (
        ingestion.audit.dropped_bytes
        == len(data) - ingestion.audit.recoverable_through_byte
    )
    assert ingestion.audit.first_nul_byte == data.index(b"\0")
    assert [record.capability_id for record in ingestion.records] == [
        "subject.one",
        "subject.two",
    ]
    orphan = ingestion.records[1]
    assert orphan.curriculum_complete is False
    assert orphan.subject_id is None
    assert orphan.subject_name is None
    assert "subject.three" not in {record.capability_id for record in ingestion.records}


def test_build_pilot_preserves_full_record_and_detects_tampering(
    tmp_path: Path,
) -> None:
    capability = _capability("subject.one", "Exact record")
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {"catalog_version": "test", "curricula": [_wrapper("S01", capability)]}
        ),
        encoding="utf-8",
    )
    ingestion = ingest_catalog(path)
    pilot = build_pilot(
        ingestion,
        [("subject.one", "Has a deterministic verifier.")],
        source_path="catalog.json",
    )

    validate_pilot(pilot)
    selected = pilot["capabilities"][0]
    assert selected["capability"] == capability
    assert selected["capability_sha256"] == canonical_sha256(capability)

    selected["capability"]["name"] = "tampered"
    with pytest.raises(CatalogError, match="hash mismatch"):
        validate_pilot(pilot)


def test_validate_pilot_rejects_audit_source_mismatch(tmp_path: Path) -> None:
    capability = _capability("subject.one")
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {"catalog_version": "test", "curricula": [_wrapper("S01", capability)]}
        ),
        encoding="utf-8",
    )
    pilot = build_pilot(
        ingest_catalog(path),
        [("subject.one", "Has a deterministic verifier.")],
        source_path="catalog.json",
    )
    pilot["catalog_audit"]["source_sha256"] = "0" * 64

    with pytest.raises(CatalogError, match="source and audit hashes differ"):
        validate_pilot(pilot)


def test_checked_in_pilot_matches_recovered_source() -> None:
    ingestion = ingest_catalog(ROOT / "catalog.json")
    pilot = load_pilot(ROOT / "data" / "pilot.json")
    source_records = {record.capability_id: record for record in ingestion.records}

    assert ingestion.audit.status == "complete"
    assert ingestion.audit.dropped_bytes == 0
    assert ingestion.audit.dropped_records_exact == 0
    assert pilot["source"]["sha256"] == ingestion.audit.source_sha256
    assert pilot["catalog_audit"] == asdict(ingestion.audit)
    assert len(pilot["capabilities"]) == 34
    assert len({item["subject_id"] for item in pilot["capabilities"]}) == 34
    assert {item["subject_id"] for item in pilot["capabilities"]} == {
        record.subject_id for record in ingestion.records
    }
    for item in pilot["capabilities"]:
        source = source_records[item["capability_id"]]
        assert source.curriculum_complete
        assert item["subject_id"] == source.subject_id
        assert item["subject_name"] == source.subject_name
        assert item["capability"] == source.capability
