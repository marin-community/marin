# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contracts for the typed Atlas license, status and catalog records."""

import pytest
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


def test_require_license_fails_closed_on_each_missing_prerequisite():
    source_id = "nemotron_sft_v3/math_v2/medium"
    with pytest.raises(ValueError):
        require_license(None, source_id, "math", "rev-1", "ingest")
    stale = receipt([evidence(("mit",))])
    stale = stale.model_copy(update={"resolver_version": "atlas-license-v1"})
    with pytest.raises(ValueError):
        require_license(stale, source_id, "math", "rev-1", "ingest")
    with pytest.raises(ValueError):
        require_license(receipt([evidence(("mit",))]), source_id, "math", "rev-2", "ingest")
    banned = receipt([evidence(("mit",))], [decision(LicenseDecisionKind.BAN, True)])
    with pytest.raises(ValueError):
        require_license(banned, source_id, "math", "rev-1", "ingest")
    failing = receipt([evidence(("mit",), restrictive_terms=True)])
    with pytest.raises(ValueError):
        require_license(failing, source_id, "math", "rev-1", "ingest")
    excepted = receipt(
        [evidence(("mit",), restrictive_terms=True)],
        [decision(LicenseDecisionKind.EXCEPTION, True)],
    )
    require_license(excepted, source_id, "math", "rev-1", "ingest")
