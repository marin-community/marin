# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the bridge's HTTP surface, against a fake finelog."""

from datetime import UTC, datetime

import pyarrow as pa
from config import BridgeConfig, ClusterTarget
from conftest import finelog_result
from finelog.errors import QueryResultTooLargeError
from server import create_app
from starlette.testclient import TestClient

# 2026-07-17T03:00:00Z and +1h, as Grafana sends them.
FROM_MS = 1_784_257_200_000
TO_MS = FROM_MS + 3_600_000

MARIN = ClusterTarget(name="marin", project="p", zone="z", instance_filter="name = finelog-marin")

# A canned result with a timestamp column, so a query returns plottable rows.
_ONE_ROW = finelog_result(t=[datetime(2026, 7, 17, 3, 0, tzinfo=UTC)], value=[1.0])


class FakeSource:
    """A MetricSource that records the SQL it is handed and replays a canned table."""

    def __init__(self, table: pa.Table | None = None, raises: Exception | None = None) -> None:
        self._table = table if table is not None else pa.table({})
        self._raises = raises
        self.queries: list[str] = []

    @property
    def target(self) -> ClusterTarget:
        return MARIN

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        self.queries.append(sql)
        if self._raises is not None:
            raise self._raises
        return self._table


def _client(source: FakeSource, **overrides) -> TestClient:
    config = BridgeConfig(
        max_rows=1000,
        cache_ttl=overrides.get("cache_ttl", 20.0),
        query_timeout_ms=5000,
    )
    return TestClient(create_app(config, {"marin": source}))


def _get(client: TestClient, sql: str, **params):
    return client.get("/marin/query", params={"sql": sql, "from": FROM_MS, "to": TO_MS, **params})


def test_query_runs_the_sql_and_returns_json_rows():
    source = FakeSource(_ONE_ROW)
    resp = _get(_client(source), 'SELECT t, value FROM "iris.task" WHERE ts >= {{from}} AND ts < {{to}}')
    assert resp.status_code == 200
    assert resp.json() == [{"t": 1_784_257_200_000, "value": 1.0}]


def test_query_substitutes_the_window_macros_before_running():
    source = FakeSource(_ONE_ROW)
    _get(_client(source), "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}")
    assert (
        source.queries[0]
        == "SELECT value FROM t WHERE ts >= TIMESTAMP '2026-07-17 03:00:00' AND ts < TIMESTAMP '2026-07-17 04:00:00'"
    )


def test_missing_sql_is_a_400():
    resp = _client(FakeSource()).get("/marin/query", params={"from": FROM_MS, "to": TO_MS})
    assert resp.status_code == 400
    assert "sql" in resp.json()["error"]


def test_a_macro_without_its_bound_is_a_400():
    resp = _client(FakeSource()).get("/marin/query", params={"sql": "SELECT 1 WHERE ts >= {{from}}"})
    assert resp.status_code == 400
    assert "no matching time bound" in resp.json()["error"]


def test_unknown_cluster_is_a_400_naming_the_valid_ones():
    resp = _client(FakeSource()).get("/nope/query", params={"sql": "SELECT 1"})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert "nope" in error and "marin" in error
    assert not error.startswith('"')


def test_oversized_result_is_a_400_telling_the_caller_what_to_do():
    resp = _get(_client(FakeSource(raises=QueryResultTooLargeError("query returned 500000 rows"))), "SELECT 1")
    assert resp.status_code == 400
    assert "narrow the time range" in resp.json()["error"]


def test_repeated_identical_panels_hit_finelog_once():
    # A shared dashboard refreshing across N viewers must not multiply through to
    # the finelog hub; Grafana's own query caching is Enterprise-only.
    source = FakeSource(_ONE_ROW)
    client = _client(source)
    first = _get(client, "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}")
    second = _get(client, "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}")
    assert first.json() == second.json()
    assert len(source.queries) == 1


def test_drifting_relative_window_still_hits_the_cache():
    # Grafana sends a relative range as absolute millis, so both edges advance a
    # little each refresh. The macros keep the SQL identical and the snapped bucket
    # keeps the key stable, so it stays one query.
    source = FakeSource(_ONE_ROW)
    client = _client(source, cache_ttl=60.0)
    sql = "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}"
    for drift in (0, 1_000, 2_500):
        _get(client, sql, **{"from": FROM_MS + drift, "to": TO_MS + drift})
    assert len(source.queries) == 1


def test_windows_further_apart_than_the_ttl_are_cached_separately():
    source = FakeSource(_ONE_ROW)
    client = _client(source, cache_ttl=20.0)
    sql = "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}"
    _get(client, sql)
    hour = 3_600_000
    _get(client, sql, **{"from": FROM_MS + hour, "to": TO_MS + hour})
    assert len(source.queries) == 2


def test_health_lists_configured_clusters():
    assert _client(FakeSource()).get("/health").json() == {"status": "ok", "clusters": ["marin"]}
