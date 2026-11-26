# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight must exercise candidate Status code rather than the preceding hosted revision."""

import requests

from infra.marina.apps.rl_data_catalog import release_check


def test_candidate_status_projection_is_checked_and_cached(monkeypatch):
    calls = []
    expected = {"sources": [{"id": "legacy", "data_restriction": "open"}]}

    class Candidate:
        def __init__(self, query):
            pass

        def get(self, path):
            calls.append(path)
            return expected

    def hosted(*args, **kwargs):
        raise AssertionError("Preflight must not read the previous hosted Status projection")

    monkeypatch.setattr(release_check, "CandidateApi", Candidate)
    monkeypatch.setattr(release_check, "marina_request", hosted)
    monkeypatch.setattr(release_check, "marina_token_provider", lambda: None)
    with requests.Session() as session:
        reader = release_check.CatalogReader("https://example.test/", session, lambda sql, parameters: {})
        assert reader.get("api/source-statuses") == expected
        assert reader.get("api/source-statuses") == expected
    assert calls == ["api/source-statuses"]


def test_candidate_taxonomy_is_served_from_candidate_code(monkeypatch):
    """Regression (R40 #50): the preceding revision served an older taxonomy without the
    backend tables, so every candidate Backends page failed in the preflight sweep."""

    def hosted(*args, **kwargs):
        raise AssertionError("Preflight must not read the previous hosted taxonomy")

    monkeypatch.setattr(release_check, "CandidateApi", lambda query: object())
    monkeypatch.setattr(release_check, "marina_request", hosted)
    monkeypatch.setattr(release_check, "marina_token_provider", lambda: None)
    with requests.Session() as session:
        reader = release_check.CatalogReader("https://example.test/", session, lambda sql, parameters: {})
        tables = reader.get("api/taxonomy")
    assert tables == release_check.candidate_taxonomy()
    assert tables["backend"] and tables["backend-reason"]
