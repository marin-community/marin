# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
    r2_account_id="a",
    r2_access_key="rk",
    r2_secret_key="rs",
    cw_endpoint="cw.example.com",
    cw_access_key="ck",
    cw_secret_key="cs",
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


def test_query_returns_result_json_with_uuid_query_id():
    result = QueryResult(
        columns=["x"],
        preview_rows=[[1], [2]],
        total_rows=5,
        truncated=True,
        result_path="gs://marin-ducky-us-east5/ducky/abc.parquet",
    )
    fake = _FakeRunner(result=result)
    client = TestClient(create_app(fake, _CONFIG))

    resp = client.post("/query", json={"sql": "SELECT * FROM range(5)"})

    assert resp.status_code == 200
    assert resp.json() == {
        "columns": ["x"],
        "rows": [[1], [2]],
        "total_rows": 5,
        "truncated": True,
        "result_path": "gs://marin-ducky-us-east5/ducky/abc.parquet",
    }
    assert fake.received_query_id is not None
    assert len(fake.received_query_id) == 32
    int(fake.received_query_id, 16)  # valid hex


def test_query_error_maps_to_400():
    fake = _FakeRunner(error=QueryError("Catalog Error: table not found"))
    client = TestClient(create_app(fake, _CONFIG))

    resp = client.post("/query", json={"sql": "SELECT * FROM nope"})

    assert resp.status_code == 400
    assert resp.json() == {"error": "Catalog Error: table not found"}


@pytest.mark.parametrize("body", [{}, {"sql": ""}, {"sql": "   "}])
def test_query_missing_sql_is_400(body):
    client = TestClient(create_app(_FakeRunner(), _CONFIG))
    resp = client.post("/query", json=body)
    assert resp.status_code == 400
    assert resp.json() == {"error": "missing 'sql'"}
