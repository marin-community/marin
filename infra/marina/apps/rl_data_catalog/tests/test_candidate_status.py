# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight must exercise candidate Status code rather than the preceding hosted revision."""

import requests
from marina.applets import validate_query

from infra.marina.apps.rl_data_catalog import release_check
from infra.marina.apps.rl_data_catalog.server import curriculum


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


def test_candidate_curriculum_sql_passes_the_read_gateway_validator():
    """Regression (R40 #57): a semicolon in a SQL comment made the read gateway refuse the
    sample-pool query as more than one statement, so the preflight could not read it."""
    for sql in (curriculum.TASKS_SQL, curriculum.AREA_SQL, curriculum.CONFIDENCE_AREA_SQL, curriculum.RELEASE_SQL):
        assert validate_query(sql) in {"SELECT", "WITH"}
