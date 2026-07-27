# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the echo API's request/response contract.

The database is faked: a stub engine returns canned rows, so these exercise wildcard
escaping, embedding modes, caller attribution, wiki persistence contracts, and 404s
without a live Postgres. PostgreSQL ranking is not covered here.
"""

import contextlib
from datetime import UTC, datetime

import app as echo
import pytest
from fastapi.testclient import TestClient


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class FakeConn:
    def __init__(self, rows, sink):
        self._rows = rows
        self._sink = sink

    def execute(self, statement, *args):
        self._sink.append(str(statement))
        return FakeResult(self._rows)

    @contextlib.contextmanager
    def _ctx(self):
        yield self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEngine:
    """Returns `rows` from every query and records compiled SQL into `statements`."""

    def __init__(self, rows):
        self.rows = rows
        self.statements: list[str] = []

    def connect(self):
        return FakeConn(self.rows, self.statements)

    def begin(self):
        return FakeConn(self.rows, self.statements)


class FakeModel:
    def __init__(self):
        self.queries: list[str] = []
        self.passages: list[str] = []

    def query_embed(self, texts):
        self.queries.extend(texts)
        return iter([[0.1, 0.2]])

    def passage_embed(self, texts):
        self.passages.extend(texts)
        return iter([[0.3, 0.4]])


def make_row(**values):
    return type("Row", (), {"_mapping": values, **values})()


@pytest.fixture
def client_with():
    def _install(rows):
        engine = FakeEngine(rows)
        model = FakeModel()
        echo.app.dependency_overrides[echo.get_engine] = lambda: engine
        echo.app.dependency_overrides[echo.get_model] = lambda: model
        return TestClient(echo.app), engine, model

    yield _install
    echo.app.dependency_overrides.clear()


def test_grep_escapes_like_wildcards():
    assert echo.escape_like("ragged_all_to_all") == "ragged\\_all\\_to\\_all"
    assert echo.escape_like("50%") == "50\\%"
    assert echo.escape_like("a\\b") == "a\\\\b"


def test_iap_caller_strips_provider_prefix():
    assert echo.iap_caller("accounts.google.com:alice@openathena.ai") == "alice@openathena.ai"
    assert echo.iap_caller(None) == "unknown"
    assert echo.iap_caller("") == "unknown"


def test_work_log_list_omits_body(client_with):
    row = make_row(id=1, at=datetime(2026, 7, 23, tzinfo=UTC), author="a", project="p", title="t", body="secret body")
    client, _, _ = client_with([row])
    entries = client.get("/work_log").json()
    assert entries == [{"id": 1, "at": "2026-07-23T00:00:00Z", "author": "a", "project": "p", "title": "t"}]
    assert "body" not in entries[0]


def test_work_log_detail_includes_body(client_with):
    row = make_row(id=1, at=datetime(2026, 7, 23, tzinfo=UTC), author="a", project="p", title="t", body="the body")
    client, _, _ = client_with([row])
    assert client.get("/work_log/1").json()["body"] == "the body"


def test_add_work_log_attributes_to_iap_caller_not_client(client_with):
    row = make_row(
        id=5, at=datetime(2026, 7, 23, tzinfo=UTC), author="bob@openathena.ai", project="p", title="t", body=None
    )
    client, _, _ = client_with([row])
    resp = client.post(
        "/work_log",
        json={"project": "p", "title": "t", "author": "somebody-else"},
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:bob@openathena.ai"},
    )
    assert resp.status_code == 201
    assert resp.json()["author"] == "bob@openathena.ai"


def test_missing_chunk_is_404(client_with):
    client, _, _ = client_with([])
    assert client.get("/chunks/999").status_code == 404


def test_activity_search_uses_query_encoder(client_with):
    row = make_row(
        id=8,
        source="github",
        kind="issue",
        date=datetime(2026, 7, 23, tzinfo=UTC),
        author="alice",
        title="Grafana dashboards",
        url="https://example.com/8",
        text="How to use Grafana.",
        score=0.04,
        distance=0.2,
        lexical_score=0.5,
    )
    client, _, model = client_with([row])
    response = client.get("/search", params={"q": "grafana"})
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Grafana dashboards"
    assert model.queries == ["grafana"]
    assert model.passages == []


def test_activity_search_rejects_whitespace_without_embedding(client_with):
    client, _, model = client_with([])
    response = client.get("/search", params={"q": "   "})
    assert response.status_code == 422
    assert model.queries == []


def test_add_wiki_embeds_title_and_body_as_passage(client_with):
    row = make_row(
        id=12,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
        updated_at=datetime(2026, 7, 27, tzinfo=UTC),
        author="agent@openathena.ai",
        title="Grafana access",
        body="Use the IAP route.",
        reference_count=0,
        score=0.0,
        distance=None,
        lexical_score=None,
    )
    client, _, model = client_with([row])
    response = client.post(
        "/wiki",
        json={"title": "  Grafana access  ", "body": "  Use the IAP route.  "},
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:agent@openathena.ai"},
    )
    assert response.status_code == 201
    assert response.json()["author"] == "agent@openathena.ai"
    assert model.passages == ["Grafana access\n\nUse the IAP route."]
    assert model.queries == []


def test_missing_wiki_entry_is_404(client_with):
    client, _, _ = client_with([])
    assert client.get("/wiki/999").status_code == 404
    assert client.post("/wiki/999/references").status_code == 404
