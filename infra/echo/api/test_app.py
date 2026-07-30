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
    def __init__(self, rows, sink, responses):
        self._rows = rows
        self._sink = sink
        self._responses = responses

    def execute(self, statement, *args):
        self._sink.append(statement)
        if self._responses:
            return FakeResult(self._responses.pop(0))
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

    def __init__(self, rows, responses=None):
        self.rows = rows
        self.responses = list(responses or [])
        self.executions: list[sqlalchemy.ClauseElement] = []

    def connect(self):
        return FakeConn(self.rows, self.executions, self.responses)

    def begin(self):
        return FakeConn(self.rows, self.executions, self.responses)


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


class FakeReranker:
    def __init__(self):
        self.queries: list[str] = []
        self.documents: list[list[str]] = []

    def rerank(self, query, documents):
        values = list(documents)
        self.queries.append(query)
        self.documents.append(values)
        return [0.0] * len(values)


def make_row(**values):
    values.setdefault("tags", [])
    return type("Row", (), {"_mapping": values, **values})()


@dataclass(frozen=True)
class ApiHarness:
    client: TestClient
    engine: FakeEngine
    model: FakeModel
    reranker: FakeReranker


@pytest.fixture
def client_with():
    def _install(rows, responses=None):
        engine = FakeEngine(rows, responses)
        model = FakeModel()
        reranker = FakeReranker()
        echo.app.dependency_overrides[echo.get_engine] = lambda: engine
        echo.app.dependency_overrides[echo.get_model] = lambda: model
        echo.app.dependency_overrides[echo.get_reranker] = lambda: reranker
        return ApiHarness(TestClient(echo.app), engine, model, reranker)

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


def test_search_configuration_keeps_discord_opt_in(client_with):
    harness = client_with([])

    configuration = harness.client.get("/api/search-configuration").json()

    assert [domain["value"] for domain in configuration["domains"]] == ["wiki", "file", "discord", "pr", "issue"]
    assert configuration["default_domains"] == ["wiki", "file", "pr", "issue"]
    assert "discord" not in configuration["default_domains"]


def test_training_log_question_uses_finelog_and_iris_query_vocabulary(client_with):
    row = make_row(
        id=8,
        source="github",
        kind="issue",
        date=None,
        author=None,
        title="Training logs",
        url="https://github.com/marin-community/marin/issues/8",
        text="Query logs from a training job.",
        score=0.04,
        distance=0.2,
        lexical_score=0.5,
    )
    harness = client_with([row])
    query = "how do i query logs from training runs?"

    echo.federated_search(harness.engine, harness.model, harness.reranker, echo.DEFAULT_CONFIG, query, ["issue"], 10)

    expanded = f"{query}\n{echo.search_config.LOG_QUERY_EXPANSION}"
    assert harness.model.queries == [expanded]
    assert harness.reranker.queries == [expanded]


def test_federated_search_classifies_github_comment_domain(client_with):
    row = make_row(
        id=8,
        source="github",
        kind="comment",
        date=datetime(2026, 7, 23, tzinfo=UTC),
        author="alice",
        title="Scheduler discussion",
        url="https://github.com/marin-community/marin/pull/7000#issuecomment-1",
        text="The scheduler can stop here.",
        score=0.04,
        distance=0.2,
        lexical_score=0.5,
    )
    harness = client_with([row])
    results = echo.federated_search(
        harness.engine,
        harness.model,
        harness.reranker,
        echo.DEFAULT_CONFIG,
        "scheduler",
        ["pr"],
        10,
    )
    assert [result.model_dump(mode="json") for result in results] == [
        {
            "id": "pr:8",
            "domain": "pr",
            "title": "Scheduler discussion",
            "subtitle": "pr · alice · 2026-07-23T00:00:00+00:00",
            "url": "https://github.com/marin-community/marin/pull/7000#issuecomment-1",
            "snippet": "The scheduler can stop here.",
            "score": 0.01639344262295082,
            "distance": 0.2,
            "lexical_score": 0.5,
            "references": [],
        }
    ]


def test_federated_file_result_names_exact_indexed_head(client_with):
    state = make_row(
        commit_sha="abcdef1234567890",
        indexed_at=datetime(2026, 7, 29, 20, tzinfo=UTC),
        completed_files=None,
        total_files=None,
        started_at=None,
    )
    file = make_row(
        id=9,
        path="lib/iris/src/iris/scheduler.py",
        title="scheduler.py",
        start_line=40,
        text="def place_gang():\n    raise FAILED_PRECONDITION\n",
        score=0.05,
        distance=0.1,
        lexical_score=4.0,
    )
    harness = client_with([], responses=[[], [state], [file]])

    response = harness.client.get(
        "/api/federated-search",
        params={"q": "FAILED_PRECONDITION", "domain": "file"},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "file:lib/iris/src/iris/scheduler.py",
            "domain": "file",
            "title": "scheduler.py",
            "subtitle": "lib/iris/src/iris/scheduler.py:41 · main@abcdef123456 · " "indexed 2026-07-29T20:00:00+00:00",
            "url": (
                "https://github.com/marin-community/marin/blob/abcdef1234567890/" "lib/iris/src/iris/scheduler.py#L41"
            ),
            "snippet": (
                "lib/iris/src/iris/scheduler.py:41 raise FAILED_PRECONDITION · "
                "lib/iris/src/iris/scheduler.py:40 def place_gang():"
            ),
            "score": 0.01639344262295082,
            "distance": 0.1,
            "lexical_score": 4.0,
            "references": [
                {
                    "line": 41,
                    "text": "raise FAILED_PRECONDITION",
                    "url": (
                        "https://github.com/marin-community/marin/blob/abcdef1234567890/"
                        "lib/iris/src/iris/scheduler.py#L41"
                    ),
                },
                {
                    "line": 40,
                    "text": "def place_gang():",
                    "url": (
                        "https://github.com/marin-community/marin/blob/abcdef1234567890/"
                        "lib/iris/src/iris/scheduler.py#L40"
                    ),
                },
            ],
        }
    ]


def test_file_summary_skips_license_boilerplate_for_filename_match():
    text = '# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"""Search Echo activity."""'

    assert echo.representative_file_lines(text, "app.py", 1) == [(4, '"""Search Echo activity."""')]


def test_file_summary_ranks_multiple_query_term_lines_with_stemming():
    text = "\n".join(
        [
            "class Cache:",
            "    pass",
            "def allocate_kv_pages():",
            "    # Caches key/value pages for generation.",
            "    return pages",
        ]
    )

    assert echo.representative_file_lines(text, "how do we handle kv caching", 20) == [
        (20, "class Cache:"),
        (22, "def allocate_kv_pages():"),
        (23, "# Caches key/value pages for generation."),
    ]


def test_prose_query_prefers_runbook_over_test_fixture():
    runbook = echo.SearchResult(
        id="file:lib/iris/OPS.md",
        domain="file",
        title="Iris Operations",
        subtitle="lib/iris/OPS.md:68",
        url="https://example.com/OPS.md",
        snippet="Restart builds and deploys your local working tree.",
        score=0.0488,
        distance=0.2355,
        lexical_score=0.21,
    )
    test_fixture = echo.SearchResult(
        id="file:infra/grafana/tests/test_k8s_source.py",
        domain="file",
        title="test_k8s_source.py",
        subtitle="infra/grafana/tests/test_k8s_source.py:106",
        url="https://example.com/test_k8s_source.py",
        snippet="def test_control_plane_uses_the_cluster_iris_namespace():",
        score=0.05386,
        distance=0.247,
        lexical_score=0.55,
    )

    ranked = sorted(
        [echo.query_oriented_result(result, "how do i deploy iris") for result in (test_fixture, runbook)],
        key=lambda result: -result.score,
    )

    assert [result.id for result in ranked] == ["file:lib/iris/OPS.md", "file:infra/grafana/tests/test_k8s_source.py"]


def test_reranker_uses_full_candidate_text_without_erasing_hybrid_rank():
    distractor = echo.SearchCandidate(
        echo.SearchResult(
            id="file:tests/test_deploy.py",
            domain="file",
            title="test_deploy.py",
            subtitle="tests/test_deploy.py:10",
            url="https://example.com/test",
            snippet="test_deploy",
            score=0.2,
            distance=0.2,
            lexical_score=0.4,
        ),
        "A deployment test fixture.",
    )
    runbook = echo.SearchCandidate(
        echo.SearchResult(
            id="file:lib/iris/OPS.md",
            domain="file",
            title="Iris Operations",
            subtitle="lib/iris/OPS.md:68",
            url="https://example.com/ops",
            snippet="Iris controller operations.",
            score=0.1,
            distance=0.2,
            lexical_score=None,
        ),
        "Restart builds and deploys the current checkout, then verifies controller health.",
    )

    class DeploymentReranker:
        def rerank(self, query, documents):
            assert query == "how do i deploy iris"
            return [float("verifies controller health" in document) for document in documents]

    ranked = echo.rerank_candidates([distractor, runbook], "how do i deploy iris", DeploymentReranker(), 2)

    assert [result.id for result in ranked] == ["file:lib/iris/OPS.md", "file:tests/test_deploy.py"]


def test_reranker_suppresses_all_candidates_below_the_quality_floor(monkeypatch):
    candidate = echo.SearchCandidate(
        echo.SearchResult(
            id="file:irrelevant.py",
            domain="file",
            title="irrelevant.py",
            subtitle="irrelevant.py:1",
            url="https://example.com/irrelevant",
            snippet="unrelated",
            score=0.1,
            distance=0.4,
            lexical_score=None,
        ),
        "This text has no relationship to the query.",
    )

    class RejectingReranker:
        def rerank(self, _query, documents):
            return [-3.0 for _ in documents]

    monkeypatch.setattr(echo.search_config, "RERANK_MAX_CANDIDATES", 2)
    candidates = [
        echo.SearchCandidate(candidate.result.model_copy(update={"id": f"file:irrelevant-{index}.py"}), candidate.text)
        for index in range(3)
    ]

    assert echo.rerank_candidates(candidates, "how do i deploy iris", RejectingReranker(), 3) == []


def test_file_artifact_reconstructs_overlapping_indexed_chunks(client_with):
    state = make_row(
        commit_sha="abcdef1234567890",
        indexed_at=datetime(2026, 7, 29, 20, tzinfo=UTC),
        completed_files=None,
        total_files=None,
        started_at=None,
    )
    first = make_row(
        path="lib/iris/OPS.md",
        title="Iris Operations",
        chunk_index=0,
        start_line=1,
        text="# Iris Operations\n\nDeploy Iris.",
    )
    second = make_row(
        path="lib/iris/OPS.md",
        title="Iris Operations",
        chunk_index=1,
        start_line=3,
        text="Deploy Iris.\nThen verify health.",
    )
    harness = client_with([], responses=[[state], [first, second]])

    response = harness.client.get("/api/repository-files/lib/iris/OPS.md")

    assert response.status_code == 200
    assert response.json() == {
        "id": "file:lib/iris/OPS.md",
        "title": "Iris Operations",
        "subtitle": "lib/iris/OPS.md · main@abcdef123456 · indexed 2026-07-29T20:00:00+00:00",
        "url": "https://github.com/marin-community/marin/blob/abcdef1234567890/lib/iris/OPS.md",
        "text": "# Iris Operations\n\nDeploy Iris.\nThen verify health.",
    }


def test_repository_index_reports_searchable_partial_build(client_with):
    state = make_row(
        commit_sha="fedcba9876543210",
        indexed_at=datetime(2026, 7, 28, 20, tzinfo=UTC),
        completed_files=70,
        total_files=180,
        started_at=datetime(2026, 7, 29, 20, tzinfo=UTC),
    )
    harness = client_with([], responses=[[state]])

    response = harness.client.get("/api/repository-index")

    assert response.status_code == 200
    assert response.json() == {
        "repository": "marin-community/marin",
        "branch": "main",
        "status": "building",
        "commit_sha": "fedcba9876543210",
        "completed_files": 70,
        "total_files": 180,
        "started_at": "2026-07-29T20:00:00Z",
        "indexed_at": "2026-07-28T20:00:00Z",
    }


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
            "tags": [" Ops ", "grafana", "ops"],
            "body": "  Use the IAP route.  ",
        },
        headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:agent@openathena.ai"},
    )
    assert response.status_code == 201
    insert_params = harness.engine.executions[-1].compile().params
    assert insert_params["author"] == "agent@openathena.ai"
    assert insert_params["use_when"] == "Use this when you need to inspect training dashboards."
    assert insert_params["tags"] == ["ops", "grafana"]
    assert harness.model.passages == [
        "Grafana access\n\nUse when: Use this when you need to inspect training dashboards."
        "\n\nTags: ops, grafana\n\nUse the IAP route."
    ]
    assert harness.model.queries == []


def test_add_wiki_requires_applicability_hint(client_with):
    harness = client_with([])
    response = harness.client.post("/api/wiki", json={"title": "Grafana access", "body": "Use the IAP route."})
    assert response.status_code == 422
    assert harness.model.passages == []


def test_add_wiki_rejects_non_slug_tag(client_with):
    harness = client_with([])
    response = harness.client.post(
        "/api/wiki",
        json={"title": "Grafana access", "use_when": "when dashboards fail", "tags": ["Needs Triage"], "body": "Body"},
    )
    assert response.status_code == 422
    assert harness.model.passages == []


def test_search_wiki_accepts_repeated_tags(client_with):
    row = make_row(
        id=12,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
        updated_at=datetime(2026, 7, 27, tzinfo=UTC),
        author="agent@openathena.ai",
        title="Iris scheduler freeze",
        use_when="when the Iris scheduler stops making progress",
        tags=["ops", "iris"],
        body="Inspect controller thread state.",
        reference_count=0,
        score=0.03,
        distance=0.2,
        lexical_score=0.4,
    )
    harness = client_with([row])
    response = harness.client.get(
        "/api/wiki/search",
        params=[("q", "scheduler freeze"), ("tag", "OPS"), ("tag", "iris")],
    )
    assert response.status_code == 200
    assert response.json()[0]["tags"] == ["ops", "iris"]


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
