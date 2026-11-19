# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contracts for the typed Atlas license, status and catalog records."""

import json

import pytest
from finestore.rl.curation import (
    INVENTORY,
    ITEMS,
    CurationCatalog,
    QualifiedSourceDeclaration,
    SnapshotSource,
    SourceDeclaration,
    register_catalog,
)
from finestore.rl.licenses import (
    LICENSE_RESOLVER_VERSION,
    LicenseBan,
    LicenseCheckVerdict,
    LicenseDecision,
    LicenseDecisionKind,
    LicenseEvidence,
    LicenseReceipt,
    check_license,
    require_license,
)
from finestore.store import DataStore

ISSUE_URL = "https://github.com/marin-community/marin/issues/11497"


def evidence(license_ids: tuple[str, ...], **overrides) -> LicenseEvidence:
    fields = {
        "url": "https://example.com/license",
        "kind": "dataset_card",
        "sha256": "a" * 64,
        "license_ids": list(license_ids),
        "quoted_lines": ["SPDX-License-Identifier: MIT"],
    }
    fields.update(overrides)
    return LicenseEvidence(**fields)


def receipt(evidence_items: list[LicenseEvidence], decisions: list[LicenseDecision] | None = None) -> LicenseReceipt:
    return LicenseReceipt(
        source_id="nemotron_sft_v3/math_v2/medium",
        dataset_id="math",
        revision="rev-1",
        resolved_at="2026-11-16T12:00:00+00:00",
        resolver_version=LICENSE_RESOLVER_VERSION,
        evidence=evidence_items,
        bundled_third_party=False,
        model_generated_outputs=False,
        decisions=decisions or [],
    )


def decision(kind: LicenseDecisionKind, approved: bool, *, revision: str = "rev-1") -> LicenseDecision:
    return LicenseDecision(
        kind=kind,
        approved=approved,
        who="coordinator",
        when="2026-11-16T12:00:00+00:00",
        source_id="nemotron_sft_v3/math_v2/medium",
        dataset_id="math",
        revision=revision,
        scope=["ingest"],
        rationale="recorded ruling",
        evidence_urls=["https://example.com/license"],
        issue_url=ISSUE_URL,
    )


def test_receipt_classification_derives_from_evidence():
    assert receipt([evidence(("mit",))]).classification.value == "permissive"
    restricted = receipt([evidence(("mit",), restrictive_terms=True)])
    assert restricted.classification.value == "restricted"
    assert receipt([evidence(("missing",))]).classification.value == "missing"
    unknown = receipt([evidence(("made-up-license",))])
    assert unknown.classification.value == "unknown"


def test_receipt_rejects_a_decision_for_another_revision():
    with pytest.raises(ValueError):
        receipt([evidence(("mit",))], [decision(LicenseDecisionKind.EXCEPTION, True, revision="rev-2")])


def test_check_license_applies_the_dataset_rules():
    allowed = check_license(receipt([evidence(("mit",))]))
    assert (allowed.verdict, allowed.rule_id.name) == (LicenseCheckVerdict.PASS, "PERMISSIVE")
    restricted = check_license(receipt([evidence(("mit",), restrictive_terms=True)]))
    assert (restricted.verdict, restricted.rule_id.name) == (LicenseCheckVerdict.FAIL, "RESTRICTED")
    missing = check_license(receipt([evidence(("missing",))]))
    assert (missing.verdict, missing.rule_id.name) == (LicenseCheckVerdict.FAIL, "MISSING")
    third_party = check_license(
        receipt([evidence(("mit",)), evidence(("missing",), unidentified_third_party_terms=True)])
    )
    assert (third_party.verdict, third_party.rule_id.name) == (LicenseCheckVerdict.ESCALATE, "THIRD_PARTY")
    conflicting = check_license(receipt([evidence(("mit",)), evidence(("made-up-license",))]))
    assert (conflicting.verdict, conflicting.rule_id.name) == (LicenseCheckVerdict.ESCALATE, "CONFLICT")


def test_check_license_reports_a_recorded_ban():
    ban = LicenseBan(
        source_id="nemotron_sft_v3/math_v2/medium",
        upstream_id="upstream",
        upstream_revision="rev-1",
        classification="restricted",
        license_ids=["cc-by-nc-4.0"],
        reason="recorded ban",
        evidence_urls=["https://example.com/license"],
        decided_at="2026-11-16T12:00:00+00:00",
        decision_by="coordinator",
        issue_url=ISSUE_URL,
    )
    check = check_license(receipt([evidence(("mit",))]), ban)
    assert (check.verdict, check.rule_id.name) == (LicenseCheckVerdict.FAIL, "BANNED")


def test_catalog_reads_legacy_and_typed_review_rows(archive_root):
    legacy = {
        "pipeline_key": "panel",
        "task_id": "old",
        "cohort": "admitted",
        "status": "reviewed",
        "verdict": "good",
        "explanation": "Recorded review",
        "evidence_ids": ["task"],
        "packet_uri": "receipt.json",
        "has_task": True,
        "omissions": [],
        "exclusion_reason": None,
    }
    typed = {
        **legacy,
        "task_id": "new",
        "verdict": "some_issues",
        "confidence": "high",
        "reference_status": "conflict",
        "defects": ["wrong_reference"],
        "repair_advisories": [{"signal": "review", "candidate_source_id": "candidate", "reason": "Check reference"}],
    }
    with DataStore.open(archive_root, writer_id="reviewer") as store:
        register_catalog(store)
        with store.transaction() as transaction:
            transaction.table(ITEMS).add(legacy)
            transaction.table(ITEMS).add(typed)
    rows = {row.task_id: row for row in CurationCatalog(archive_root).items("panel")}
    assert rows["old"].confidence is None
    assert rows["old"].defects == []
    assert rows["old"].repair_advisories == []
    assert rows["new"].confidence == "high"
    assert rows["new"].reference_status == "conflict"
    assert rows["new"].defects == ["wrong_reference"]
    assert rows["new"].repair_advisories[0].reason == "Check reference"


def legacy_source():
    return {
        "id": "math",
        "name": "math",
        "display_name": "Math",
        "origin": "native",
        "family": "math",
        "tags": [],
        "task_count": 3,
        "notes": "",
        "dataset_id": None,
        "dataset_revision": None,
        "url": None,
        "type": "RLVR",
        "turns": "Single-turn",
        "is_benchmark": False,
        "kind": "Dataset",
        "status": "Available",
        "pipeline": None,
    }


def test_legacy_source_writer_and_catalog_reader_keep_their_contract(archive_root):
    declaration = SourceDeclaration.model_validate(legacy_source())
    source = SnapshotSource(
        snapshot_id="legacy",
        id=declaration.id,
        declaration=declaration,
        assessment_key=None,
        latest_attempt_key=None,
        latest_attempt_status="skipped",
        reason=None,
    )
    with DataStore.open(archive_root, writer_id="legacy") as store:
        register_catalog(store)
        with store.transaction() as transaction:
            transaction.table(INVENTORY).add(source.model_dump(mode="json"))
    restored = CurationCatalog(archive_root).sources("legacy")
    assert [item.declaration.model_dump(mode="json") for item in restored] == [declaration.model_dump(mode="json")]
    assert declaration.daytona_safety == "unassessed"


def qualified_source():
    return {
        **legacy_source(),
        "id": "Marin:math",
        "origin": "Marin",
        "build_path": "Pipeline",
        "upstream_dataset": "math",
        "upstream_name": "Math",
        "dataset_id": "math",
        "dataset_revision": "revision-1",
        "daytona_safety": None,
        "pipeline": {"name": "math-v1", "source": "math", "revision": "revision-1", "version": "1", "files": []},
        "current_release": {"repo": "math", "revision": "revision-1", "url": "s3://bucket/math"},
        "within_source_dedup": {
            "raw_count": 4,
            "unique_count": 3,
            "within_source_duplicates": 1,
            "source_dataset_id": "math",
            "source_revision": "revision-1",
            "converter_revision": "converter-1",
            "converter_version": "1",
            "drop_manifest_uri": "s3://bucket/drops.json",
            "evidence_url": "s3://bucket/dedup.json",
        },
    }


def test_qualified_source_preserves_current_input_and_rejects_stale_evidence():
    row = qualified_source()
    declaration = QualifiedSourceDeclaration.model_validate_json(json.dumps(row))
    assert declaration.current_release.revision == "revision-1"
    assert declaration.within_source_dedup.unique_count == 3
    for change in ({"task_count": 4}, {"dataset_revision": "revision-2"}, {"display_name": "Unrelated"}):
        with pytest.raises(ValueError):
            QualifiedSourceDeclaration.model_validate({**row, **change})


def test_require_license_applies_a_registry_ban():
    source_id = "nemotron_sft_v3/math_v2/medium"
    ban = LicenseBan(
        source_id=source_id,
        upstream_id="upstream",
        upstream_revision="rev-1",
        classification="restricted",
        license_ids=["cc-by-nc-4.0"],
        reason="recorded registry ban",
        evidence_urls=["https://example.com/license"],
        decided_at="2026-11-16T12:00:00+00:00",
        decision_by="coordinator",
        issue_url=ISSUE_URL,
    )
    with pytest.raises(ValueError, match="blocks ingest and admission"):
        require_license(receipt([evidence(("mit",))]), source_id, "math", "rev-1", "ingest", ban=ban)


def test_require_license_fails_closed_on_each_missing_prerequisite():
    source_id = "nemotron_sft_v3/math_v2/medium"
    with pytest.raises(ValueError):
        require_license(None, source_id, "math", "rev-1", "ingest", ban=None)
    stale = receipt([evidence(("mit",))])
    stale = stale.model_copy(update={"resolver_version": "atlas-license-v1"})
    with pytest.raises(ValueError):
        require_license(stale, source_id, "math", "rev-1", "ingest", ban=None)
    with pytest.raises(ValueError):
        require_license(receipt([evidence(("mit",))]), source_id, "math", "rev-2", "ingest", ban=None)
    banned = receipt([evidence(("mit",))], [decision(LicenseDecisionKind.BAN, True)])
    with pytest.raises(ValueError):
        require_license(banned, source_id, "math", "rev-1", "ingest", ban=None)
    failing = receipt([evidence(("mit",), restrictive_terms=True)])
    with pytest.raises(ValueError):
        require_license(failing, source_id, "math", "rev-1", "ingest", ban=None)
    excepted = receipt(
        [evidence(("mit",), restrictive_terms=True)],
        [decision(LicenseDecisionKind.EXCEPTION, True)],
    )
    require_license(excepted, source_id, "math", "rev-1", "ingest", ban=None)
