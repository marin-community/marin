# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the grafana bridge: its HTTP surface over a fake finelog, and the
cache's coalescing and eviction contract."""

import threading
from datetime import UTC, datetime

import pyarrow as pa
from cache import TtlCache
from config import ClusterTarget
from conftest import bridge_config
from finelog.errors import QueryResultTooLargeError
from github_source import GithubSource
from k8s_source import K8sFleet
from server import create_app
from starlette.testclient import TestClient

# 2026-07-17T03:00:00Z and +1h, as Grafana sends them.
FROM_MS = 1_784_257_200_000
TO_MS = FROM_MS + 3_600_000
MARIN = ClusterTarget(
    name="marin", project="p", zone="z", instance_filter="name = finelog-marin", controller_filter="labels.x=true"
)


def finelog_result(**columns: list) -> pa.Table:
    """An Arrow table shaped like a finelog result; types are inferred from the values."""
    return pa.table(dict(columns))


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


def _client(source: FakeSource, cache_ttl: float = 20.0) -> TestClient:
    github = GithubSource(token=None, timeout=5.0)
    return TestClient(create_app(bridge_config(cache_ttl), {"marin": source}, {}, github, K8sFleet(())))


def _get(client: TestClient, sql: str, **params):
    return client.get("/finelog/marin/query", params={"sql": sql, "from": FROM_MS, "to": TO_MS, **params})


def test_query_returns_json_rows_with_millis_timestamps():
    resp = _get(_client(FakeSource(_ONE_ROW)), 'SELECT t, value FROM "iris.task" WHERE ts >= {{from}} AND ts < {{to}}')
    assert resp.status_code == 200
    assert resp.json() == [{"t": 1_784_257_200_000, "value": 1.0}]


def test_query_substitutes_window_macros_before_running():
    source = FakeSource(_ONE_ROW)
    _get(_client(source), "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}")
    assert source.queries == [
        "SELECT value FROM t WHERE ts >= TIMESTAMP '2026-07-17 03:00:00' AND ts < TIMESTAMP '2026-07-17 04:00:00'"
    ]


def test_missing_sql_is_a_400():
    resp = _client(FakeSource()).get("/finelog/marin/query", params={"from": FROM_MS, "to": TO_MS})
    assert resp.status_code == 400
    assert "sql" in resp.json()["error"]


def test_macro_without_its_bound_is_a_400():
    resp = _client(FakeSource()).get("/finelog/marin/query", params={"sql": "SELECT 1 WHERE ts >= {{from}}"})
    assert resp.status_code == 400
    assert "no matching time bound" in resp.json()["error"]


def test_unknown_cluster_is_a_400_naming_the_valid_ones():
    resp = _client(FakeSource()).get("/finelog/nope/query", params={"sql": "SELECT 1"})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert "nope" in error and "marin" in error


def test_oversized_result_is_a_400_with_guidance():
    resp = _get(_client(FakeSource(raises=QueryResultTooLargeError("query returned 500000 rows"))), "SELECT 1")
    assert resp.status_code == 400
    assert "narrow the time range" in resp.json()["error"]


def test_repeated_identical_panels_hit_finelog_once():
    source = FakeSource(_ONE_ROW)
    client = _client(source)
    sql = "SELECT value FROM t WHERE ts >= {{from}} AND ts < {{to}}"
    assert _get(client, sql).json() == _get(client, sql).json()
    assert len(source.queries) == 1


def test_relative_window_drifting_within_the_ttl_stays_one_query():
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
    hour = 3_600_000
    _get(client, sql)
    _get(client, sql, **{"from": FROM_MS + hour, "to": TO_MS + hour})
    assert len(source.queries) == 2


def test_json_labels_flatten_into_columns():
    source = FakeSource(finelog_result(value=[3.0], labels=['{"region": "us-east5", "scope": "pool"}']))
    assert _get(_client(source), "SELECT value, labels FROM t").json() == [
        {"value": 3.0, "label_region": "us-east5", "label_scope": "pool"}
    ]


def test_native_map_labels_flatten_into_columns():
    # A native Map<Utf8,Utf8> column (telltale metrics) arrives as list[(k, v)].
    table = pa.table(
        {"value": [3.0], "labels": [[("region", "us-east5"), ("scope", "pool")]]},
        schema=pa.schema([("value", pa.float64()), ("labels", pa.map_(pa.string(), pa.string()))]),
    )
    source = FakeSource(table)
    assert _get(_client(source), "SELECT value, labels FROM t").json() == [
        {"value": 3.0, "label_region": "us-east5", "label_scope": "pool"}
    ]


def test_unparseable_labels_cell_keeps_the_row():
    # One malformed cell is schema drift; the panel still gets its row.
    source = FakeSource(finelog_result(value=[1.0], labels=["{not json"]))
    assert _get(_client(source), "SELECT value, labels FROM t").json() == [{"value": 1.0, "labels": "{not json"}]


def test_health_lists_configured_clusters():
    assert _client(FakeSource()).get("/health").json() == {"status": "ok", "clusters": ["marin"]}


def test_cache_coalesces_concurrent_misses_on_one_key():
    # N callers racing a cold key compute once. Pin the order: the first caller is
    # inside compute, holding the key lock, before the rest start.
    cache: TtlCache[int] = TtlCache(ttl=60.0)
    computing = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    results: list[int] = []

    def compute():
        calls.append(1)
        computing.set()
        release.wait(timeout=5)
        return 7

    def worker():
        results.append(cache.get_or_compute("k", compute))

    first = threading.Thread(target=worker)
    first.start()
    assert computing.wait(timeout=5), "first caller never entered compute"
    others = [threading.Thread(target=worker) for _ in range(3)]
    for t in others:
        t.start()
    release.set()
    for t in [first, *others]:
        t.join(timeout=10)

    assert len(calls) == 1
    assert results == [7, 7, 7, 7]


def test_cache_prunes_expired_entries_on_write():
    # Keys embed a rotating time bucket, so an insert-only cache grows without bound
    # on a long-lived process. At ttl=0 every entry is stale on arrival.
    cache: TtlCache[int] = TtlCache(ttl=0.0)
    for i in range(50):
        cache.get_or_compute(f"bucket-{i}", lambda i=i: i)
    assert len(cache) == 0
