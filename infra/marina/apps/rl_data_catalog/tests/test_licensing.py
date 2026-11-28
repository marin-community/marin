# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned ruling registry loader and source ban queries."""

import hashlib
from datetime import UTC, datetime

import pytest
from finestore.rl.licenses import LicenseBan, LicenseReceipt, LicenseSourcePin

from infra.marina.apps.rl_data_catalog.server import licensing


def ban(source_id: str, upstream_id: str = "example/test-dataset") -> LicenseBan:
    return LicenseBan(
        source_id=source_id,
        upstream_id=upstream_id,
        upstream_revision="a" * 40,
        classification="unknown",
        license_ids=["unknown"],
        reason="Upstream task-data permission has not been established.",
        evidence_urls=["https://example.test/terms"],
        decided_at="2026-11-15T17:00:00+00:00",
        decision_by="user via coordinator",
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )


def resolved_pin(dataset_id: str) -> LicenseSourcePin:
    return LicenseSourcePin(
        dataset_id=dataset_id,
        revision="1",
        url=f"https://example.test/{dataset_id}",
        declaration=license_evidence(dataset_id, "MIT"),
    )


def license_evidence(dataset_id: str, content: str) -> dict:
    return {
        "url": f"https://example.test/{dataset_id}/LICENSE",
        "kind": "license_file",
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "license_ids": ["MIT"],
        "quoted_lines": [content],
    }


def receipt(source_id: str, dataset_id: str, resolved: LicenseSourcePin | None) -> LicenseReceipt:
    return LicenseReceipt(
        source_id=source_id,
        dataset_id=dataset_id,
        revision="1",
        resolved_at=datetime.now(UTC).isoformat(),
        resolver_version="fixture",
        evidence=[license_evidence(dataset_id, "MIT")],
        bundled_third_party=False,
        model_generated_outputs=False,
        resolved_source=resolved,
    )


@pytest.fixture
def rulings(monkeypatch):
    """Stage a ruling document behind the loader's registry fetch."""

    def stage(document) -> None:
        monkeypatch.setattr(licensing, "read_registry_json", lambda _reference: document)
        licensing.license_bans.cache_clear()

    yield stage
    licensing.license_bans.cache_clear()


def test_loader_reads_the_pinned_ruling_document(rulings):
    rulings({"version": "fixture-1", "bans": [ban("Banned:source").model_dump(mode="json")]})
    bans = licensing.license_bans()
    assert set(bans) == {"Banned:source"}
    assert bans["Banned:source"].reason == "Upstream task-data permission has not been established."


def test_loader_refuses_duplicate_source_ids(rulings):
    rulings({"version": "fixture-1", "bans": [ban("Banned:source").model_dump(mode="json")] * 2})
    with pytest.raises(ValueError, match="duplicate source IDs"):
        licensing.license_bans()


def test_source_ban_matches_the_source_then_its_upstream(rulings):
    rulings(
        {
            "version": "fixture-1",
            "bans": [ban("Banned:source", "example/renamed-dataset").model_dump(mode="json")],
        }
    )
    assert licensing.source_license_ban("Other:source", "example/test-dataset") is None
    assert licensing.source_license_ban("Other:source", "example/renamed-dataset") is not None
    with pytest.raises(ValueError, match="License banned"):
        licensing.require_unbanned("Other:source", "example/renamed-dataset")


def test_source_ban_follows_the_receipt_resolved_source(rulings):
    rulings(
        {
            "version": "fixture-1",
            "bans": [ban("Banned:source", "example/renamed-dataset").model_dump(mode="json")],
        }
    )
    typed = receipt("Other:source", "example/test-dataset", resolved_pin("example/renamed-dataset"))
    assert licensing.source_license_ban("Other:source", "example/test-dataset", typed) is not None


def test_committed_reference_resolves_the_published_rulings():
    """The pinned revision still carries the published ruling bytes (network)."""
    document = licensing.license_bans()
    assert document
    assert all(ban.issue_url.startswith("https://github.com/marin-community/marin/issues/") for ban in document.values())
