# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the echo API's request/response contract.

The database is faked: a stub engine returns canned rows, so these exercise wildcard
escaping, embedding modes, caller attribution, wiki persistence contracts, and 404s
without a live Postgres. PostgreSQL ranking is not covered here.
"""

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime

import app as echo
import pytest
import sqlalchemy
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
        self._sink.append(statement)
        return FakeResult(self._rows)

    @contextlib.contextmanager
    def _ctx(self):
        yield self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEngine:
    """Returns `rows` from every query and records executed SQL expressions."""

    def __init__(self, rows):
        self.rows = rows
        self.executions: list[sqlalchemy.ClauseElement] = []

    def connect(self):
        return FakeConn(self.rows, self.executions)

    def begin(self):
        return FakeConn(self.rows, self.executions)


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


@dataclass(frozen=True)
class ApiHarness:
    client: TestClient
    engine: FakeEngine
    model: FakeModel


@pytest.fixture
def client_with():
    def _install(rows):
        engine = FakeEngine(rows)
        model = FakeModel()
        echo.app.dependency_overrides[echo.get_engine] = lambda: engine
        echo.app.dependency_overrides[echo.get_model] = lambda: model
        return ApiHarness(TestClient(echo.app), engine, model)

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
    harness = client_with([row])
    entries = harness.client.get("/api/work_log").json()
    assert entries == [{"id": 1, "at": "2026-07-23T00:00:00Z", "author": "a", "project": "p", "title": "t"}]
    assert "body" not in entries[0]


def test_work_log_detail_includes_body(client_with):
    row = make_row(id=1, at=datetime(2026, 7, 23, tzinfo=UTC), author="a", project="p", title="t", body="the body")
    harness = client_with([row])
    assert harness.client.get("/api/work_log/1").json()["body"] == "the body"


def test_add_work_log_attributes_to_iap_caller_not_client(client_with):
    row = make_row(
        id=5, at=datetime(2026, 7, 23, tzinfo=UTC), author="bob@openathena.ai", project="p", title="t", body=None
    )
    harness = client_with([row])
    resp = harness.client.post(
        "/api/work_log",
        json={"project": "p", "title": "t", "author": "somebody-else"},
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:bob@openathena.ai"},
    )
    assert resp.status_code == 201
    assert resp.json()["author"] == "bob@openathena.ai"


def test_missing_chunk_is_404(client_with):
    harness = client_with([])
    assert harness.client.get("/api/chunks/999").status_code == 404


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
    harness = client_with([row])
    response = harness.client.get("/api/search", params={"q": "grafana"})
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Grafana dashboards"
    assert harness.model.queries == ["grafana"]
    assert harness.model.passages == []


def test_activity_search_rejects_whitespace_without_embedding(client_with):
    harness = client_with([])
    response = harness.client.get("/api/search", params={"q": "   "})
    assert response.status_code == 422
    assert harness.model.queries == []


def test_add_wiki_embeds_applicability_hint_and_body_as_passage(client_with):
    row = make_row(
        id=12,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
        updated_at=datetime(2026, 7, 27, tzinfo=UTC),
        author="agent@openathena.ai",
        title="Grafana access",
        use_when="Use this when you need to inspect training dashboards.",
        body="Use the IAP route.",
        reference_count=0,
        score=0.0,
        distance=None,
        lexical_score=None,
    )
    harness = client_with([row])
    response = harness.client.post(
        "/api/wiki",
        json={
            "title": "  Grafana access  ",
            "use_when": "  Use this when you need to inspect training dashboards.  ",
            "body": "  Use the IAP route.  ",
        },
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:agent@openathena.ai"},
    )
    assert response.status_code == 201
    insert_params = harness.engine.executions[-1].compile().params
    assert insert_params["author"] == "agent@openathena.ai"
    assert insert_params["use_when"] == "Use this when you need to inspect training dashboards."
    assert harness.model.passages == [
        "Grafana access\n\nUse when: Use this when you need to inspect training dashboards.\n\nUse the IAP route."
    ]
    assert harness.model.queries == []


def test_add_wiki_requires_applicability_hint(client_with):
    harness = client_with([])
    response = harness.client.post("/api/wiki", json={"title": "Grafana access", "body": "Use the IAP route."})
    assert response.status_code == 422
    assert harness.model.passages == []


def test_update_wiki_re_embeds_and_keeps_author(client_with):
    row = make_row(
        id=12,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
        updated_at=datetime(2026, 7, 28, tzinfo=UTC),
        author="original@openathena.ai",
        title="Grafana access",
        use_when="Use this when inspecting dashboards.",
        body="Use the IAP route via grafana.oa.dev.",
        reference_count=3,
        score=0.0,
        distance=None,
        lexical_score=None,
    )
    harness = client_with([row])
    response = harness.client.put(
        "/api/wiki/12",
        json={
            "title": "  Grafana access  ",
            "use_when": "  Use this when inspecting dashboards.  ",
            "body": "  Use the IAP route via grafana.oa.dev.  ",
        },
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:editor@openathena.ai"},
    )
    assert response.status_code == 200
    update_params = harness.engine.executions[-1].compile().params
    # A PUT re-embeds the passage and never rewrites the original author.
    assert "author" not in update_params
    assert update_params["body"] == "Use the IAP route via grafana.oa.dev."
    assert harness.model.passages == [
        "Grafana access\n\nUse when: Use this when inspecting dashboards.\n\nUse the IAP route via grafana.oa.dev."
    ]


def test_update_missing_wiki_is_404(client_with):
    harness = client_with([])
    response = harness.client.put("/api/wiki/999", json={"title": "t", "use_when": "when", "body": "b"})
    assert response.status_code == 404


def test_update_wiki_rejects_blank_field(client_with):
    harness = client_with([])
    response = harness.client.put("/api/wiki/1", json={"title": "t", "use_when": "   ", "body": "b"})
    assert response.status_code == 422
    assert harness.model.passages == []


def test_missing_wiki_entry_is_404(client_with):
    harness = client_with([])
    assert harness.client.get("/api/wiki/999").status_code == 404
    assert harness.client.post("/api/wiki/999/references").status_code == 404
