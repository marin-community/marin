# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import Future

import pytest
from ducky.config import DuckyConfig
from ducky.runner import QueryError, QueryResult
from ducky.server import QueryManager, create_app
from starlette.testclient import TestClient

_CONFIG = DuckyConfig(
    scratch_bucket="gs://marin-ducky-us-east5",
    gcs_hmac_key_id="k",
    gcs_hmac_secret="s",
    result_ttl_days=7,
)


class _InlineExecutor:
    """Runs submitted work synchronously, so a query finishes during ``submit`` — the
    test reads ``/result`` once with no polling or ``time.sleep`` (per root TESTING.md)."""

    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:  # mirror ThreadPoolExecutor: capture into the future
            future.set_exception(e)
        return future

    def shutdown(self, wait=True, **kwargs):
        pass


class _FakeRunner:
    """Stands in for QueryRunner: records the query_id and returns a canned result."""

    def __init__(self, result: QueryResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.received_query_id: str | None = None

    def run_query(self, sql: str, query_id: str) -> QueryResult:
        self.received_query_id = query_id
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _client(runner) -> TestClient:
    return TestClient(create_app(runner, _CONFIG, executor=_InlineExecutor()))


def _run(client: TestClient, sql: str) -> dict:
    """Submit and read the result. The inline executor finishes the query during submit."""
    query_id = client.post("/query", json={"sql": sql}).json()["query_id"]
    payload = client.get(f"/result/{query_id}").json()
    assert payload["status"] != "running"  # inline executor → already terminal
    return payload


def test_health_is_public():
    resp = _client(_FakeRunner()).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def _built_dist(tmp_path):
    dist = tmp_path / "dist"
    (dist / "static").mkdir(parents=True)
    (dist / "index.html").write_text('<!doctype html><base href="/" /><div id="app"></div>', encoding="utf-8")
    return dist


def test_index_serves_spa_and_rewrites_base(tmp_path, monkeypatch):
    monkeypatch.setenv("DUCKY_DASHBOARD_DIST", str(_built_dist(tmp_path)))
    resp = _client(_FakeRunner()).get("/", headers={"X-Forwarded-Prefix": "/proxy/ducky"})
    assert resp.status_code == 200
    assert 'id="app"' in resp.text
    assert '<base href="/proxy/ducky/"' in resp.text  # rewritten for the proxy sub-path


def test_index_without_prefix_keeps_root_base(tmp_path, monkeypatch):
    monkeypatch.setenv("DUCKY_DASHBOARD_DIST", str(_built_dist(tmp_path)))
    resp = _client(_FakeRunner()).get("/")
    assert '<base href="/"' in resp.text


def test_index_not_built_returns_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("DUCKY_DASHBOARD_DIST", str(tmp_path / "absent"))
    resp = _client(_FakeRunner()).get("/")
    assert resp.status_code == 503
    assert "not built" in resp.text.lower()


def test_api_config_returns_ttl():
    resp = _client(_FakeRunner()).get("/api/config")
    assert resp.status_code == 200
    assert resp.json() == {"result_ttl_days": 7}


def test_query_accepts_and_returns_uuid_query_id():
    fake = _FakeRunner(result=QueryResult(["x"], [[1]], 1, False, "gs://b/ducky/a.parquet", 12, 345))
    resp = _client(fake).post("/query", json={"sql": "SELECT 1"})

    assert resp.status_code == 202
    query_id = resp.json()["query_id"]
    assert len(query_id) == 32
    int(query_id, 16)  # valid hex


def test_query_result_delivered_via_result_endpoint():
    result = QueryResult(["x"], [[1], [2]], 5, True, "gs://marin-ducky-us-east5/ducky/abc.parquet", 1234, 5678)
    fake = _FakeRunner(result=result)

    payload = _run(_client(fake), "SELECT * FROM range(5)")

    assert payload == {
        "status": "done",
        "columns": ["x"],
        "rows": [[1], [2]],
        "total_rows": 5,
        "truncated": True,
        "result_path": "gs://marin-ducky-us-east5/ducky/abc.parquet",
        "cached": False,
        "elapsed_ms": 1234,
        "result_bytes": 5678,
    }
    assert fake.received_query_id is not None


def test_identical_sql_served_from_cache():
    """A second run of the same SQL is served from cache without re-executing."""

    class _CountingRunner(_FakeRunner):
        def __init__(self, result):
            super().__init__(result=result)
            self.calls = 0

        def run_query(self, sql, query_id):
            self.calls += 1
            return super().run_query(sql, query_id)

    runner = _CountingRunner(QueryResult(["x"], [[1]], 1, False, "gs://b/ducky/first.parquet", 50, 99))
    client = _client(runner)

    first = _run(client, "SELECT 1")
    second = _run(client, "SELECT 1")

    assert runner.calls == 1  # executed once, second served from cache
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["result_path"] == "gs://b/ducky/first.parquet"  # reuses the spilled file


def test_query_error_surfaces_in_result():
    fake = _FakeRunner(error=QueryError("Catalog Error: table not found"))
    payload = _run(_client(fake), "SELECT * FROM nope")
    assert payload == {"status": "error", "error": "Catalog Error: table not found"}


def test_unknown_query_id_is_404():
    resp = _client(_FakeRunner()).get("/result/deadbeef")
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown query_id"}


@pytest.mark.parametrize("body", [{}, {"sql": ""}, {"sql": "   "}])
def test_query_missing_sql_is_400(body):
    resp = _client(_FakeRunner()).post("/query", json=body)
    assert resp.status_code == 400
    assert resp.json() == {"error": "missing 'sql'"}


def test_cache_entry_expires_after_ttl():
    """A cached result older than the TTL is dropped (its spilled parquet may be gone)."""
    manager = QueryManager(_FakeRunner(), executor=_InlineExecutor(), cache_ttl=10)
    result = QueryResult(["a"], [[1]], 1, False, "gs://b/x.parquet", 1, 1)
    manager._cache["SELECT 1"] = (result, time.monotonic() - 100)  # stale
    manager._cache["SELECT 2"] = (result, time.monotonic())  # fresh
    assert manager._cached_result("SELECT 1") is None
    assert manager._cached_result("SELECT 2") is result
