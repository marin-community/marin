# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from ducky.config import DuckyConfig
from ducky.runner import QueryError, QueryResult
from ducky.server import create_app
from starlette.testclient import TestClient

_CONFIG = DuckyConfig(
    region="us-east5",
    scratch_bucket="gs://marin-ducky-us-east5",
    gcs_hmac_key_id="k",
    gcs_hmac_secret="s",
    result_ttl_days=7,
)


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


def _poll(client: TestClient, query_id: str, timeout: float = 3.0) -> dict:
    """Poll /result until the query leaves the 'running' state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/result/{query_id}").json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.01)
    raise AssertionError(f"query {query_id} still running after {timeout}s")


def test_health_is_public():
    client = TestClient(create_app(_FakeRunner(), _CONFIG))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_index_serves_form():
    client = TestClient(create_app(_FakeRunner(), _CONFIG))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<textarea" in resp.text


def test_query_accepts_and_returns_uuid_query_id():
    fake = _FakeRunner(result=QueryResult(["x"], [[1]], 1, False, "gs://b/ducky/a.parquet"))
    client = TestClient(create_app(fake, _CONFIG))

    resp = client.post("/query", json={"sql": "SELECT 1"})

    assert resp.status_code == 202
    query_id = resp.json()["query_id"]
    assert len(query_id) == 32
    int(query_id, 16)  # valid hex


def test_query_result_delivered_via_polling():
    result = QueryResult(["x"], [[1], [2]], 5, True, "gs://marin-ducky-us-east5/ducky/abc.parquet")
    fake = _FakeRunner(result=result)
    client = TestClient(create_app(fake, _CONFIG))

    query_id = client.post("/query", json={"sql": "SELECT * FROM range(5)"}).json()["query_id"]
    payload = _poll(client, query_id)

    assert payload == {
        "status": "done",
        "columns": ["x"],
        "rows": [[1], [2]],
        "total_rows": 5,
        "truncated": True,
        "result_path": "gs://marin-ducky-us-east5/ducky/abc.parquet",
        "cached": False,
    }
    assert fake.received_query_id == query_id


def test_identical_sql_served_from_cache():
    """A second run of the same SQL is served from cache without re-executing."""

    class _CountingRunner(_FakeRunner):
        def __init__(self, result):
            super().__init__(result=result)
            self.calls = 0

        def run_query(self, sql, query_id):
            self.calls += 1
            return super().run_query(sql, query_id)

    result = QueryResult(["x"], [[1]], 1, False, "gs://b/ducky/first.parquet")
    runner = _CountingRunner(result)
    client = TestClient(create_app(runner, _CONFIG))

    first = _poll(client, client.post("/query", json={"sql": "SELECT 1"}).json()["query_id"])
    second = _poll(client, client.post("/query", json={"sql": "SELECT 1"}).json()["query_id"])

    assert runner.calls == 1  # executed once, second served from cache
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["result_path"] == "gs://b/ducky/first.parquet"  # reuses the spilled file


def test_query_error_surfaces_in_result():
    fake = _FakeRunner(error=QueryError("Catalog Error: table not found"))
    client = TestClient(create_app(fake, _CONFIG))

    query_id = client.post("/query", json={"sql": "SELECT * FROM nope"}).json()["query_id"]
    payload = _poll(client, query_id)

    assert payload == {"status": "error", "error": "Catalog Error: table not found"}


def test_unknown_query_id_is_404():
    client = TestClient(create_app(_FakeRunner(), _CONFIG))
    resp = client.get("/result/deadbeef")
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown query_id"}


@pytest.mark.parametrize("body", [{}, {"sql": ""}, {"sql": "   "}])
def test_query_missing_sql_is_400(body):
    client = TestClient(create_app(_FakeRunner(), _CONFIG))
    resp = client.post("/query", json=body)
    assert resp.status_code == 400
    assert resp.json() == {"error": "missing 'sql'"}
