# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the bridge's HTTP surface, against a fake finelog."""

import pyarrow as pa
import pytest
from config import BridgeConfig, ClusterTarget
from conftest import result_table as _rows
from finelog.errors import QueryResultTooLargeError
from server import create_app
from starlette.testclient import TestClient

# 2026-07-17T03:00:00Z and +1h, as Grafana sends them.
FROM_MS = 1_784_257_200_000
TO_MS = FROM_MS + 3_600_000


MARIN = ClusterTarget(name="marin", project="p", zone="z", instance_filter="name = finelog-marin")


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


def _client(source: FakeSource, **overrides) -> tuple[TestClient, FakeSource]:
    config = BridgeConfig(
        max_rows=1000,
        cache_ttl=overrides.get("cache_ttl", 20.0),
        max_window_hours=overrides.get("max_window_hours", 168.0),
        query_timeout_ms=5000,
    )
    return TestClient(create_app(config, {"marin": source})), source


def test_series_returns_long_rows_grouped_by_label():
    source = FakeSource(
        _rows(
            [
                ("worker_healthy", 3.0, {"region": "us-east5"}, 1_784_257_200_000_000),
                ("worker_healthy", 5.0, {"region": "us-central2"}, 1_784_257_200_000_000),
            ]
        )
    )
    client, _ = _client(source)
    resp = client.get(
        "/marin/series", params={"metric": "worker_healthy", "from": FROM_MS, "to": TO_MS, "group_by": "region"}
    )
    assert resp.status_code == 200
    assert resp.json() == [
        {"time": 1_784_257_200_000, "series": "us-east5", "value": 3.0},
        {"time": 1_784_257_200_000, "series": "us-central2", "value": 5.0},
    ]


def test_label_filter_selects_the_fleet_rollup():
    # probes emits a fleet rollup and per-region rows under one metric name;
    # DataFusion cannot slice the JSON labels, so the bridge does it.
    source = FakeSource(
        _rows(
            [
                ("worker_healthy", 8.0, {"scope": "fleet"}, 1_784_257_200_000_000),
                ("worker_healthy", 3.0, {"region": "us-east5"}, 1_784_257_200_000_000),
            ]
        )
    )
    client, _ = _client(source)
    resp = client.get(
        "/marin/series",
        params={"metric": "worker_healthy", "from": FROM_MS, "to": TO_MS, "label.scope": "fleet"},
    )
    assert [r["value"] for r in resp.json()] == [8.0]


def test_accepts_iso_instants_as_well_as_grafana_millis():
    client, source = _client(FakeSource(_rows([])))
    resp = client.get(
        "/marin/series",
        params={"metric": "m", "from": "2026-07-17T03:00:00Z", "to": "2026-07-17T04:00:00Z"},
    )
    assert resp.status_code == 200
    assert "TIMESTAMP '2026-07-17 03:00:00'" in source.queries[0]


def test_unknown_cluster_is_a_400_naming_the_valid_ones():
    client, _ = _client(FakeSource())
    resp = client.get("/nope/series", params={"metric": "m", "from": FROM_MS, "to": TO_MS})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert "nope" in error and "marin" in error
    # A plain message, not a KeyError repr — that would arrive double-quoted.
    assert not error.startswith('"')


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"from": FROM_MS, "to": TO_MS}, "metric"),
        ({"metric": "m", "to": TO_MS}, "from"),
        ({"metric": "m", "from": TO_MS, "to": FROM_MS}, "must be after"),
        ({"metric": "m; DROP TABLE x --", "from": FROM_MS, "to": TO_MS}, "bare identifier"),
    ],
)
def test_malformed_panel_queries_are_400_not_500(params, expected):
    client, _ = _client(FakeSource(_rows([])))
    resp = client.get("/marin/series", params=params)
    assert resp.status_code == 400
    assert expected in resp.json()["error"]


def test_schema_drift_is_a_server_error_not_a_caller_error():
    # A result missing the columns we selected means finelog's rows changed under
    # us. Reporting our own data bug as the panel's mistake would send whoever
    # debugs it looking in the wrong place.
    client, _ = _client(FakeSource(pa.table({"metric": ["m"]})))
    with pytest.raises(ValueError, match="missing columns"):
        client.get("/marin/series", params={"metric": "m", "from": FROM_MS, "to": TO_MS})


def test_window_beyond_the_limit_is_refused_before_querying():
    client, source = _client(FakeSource(_rows([])), max_window_hours=1.0)
    resp = client.get("/marin/series", params={"metric": "m", "from": FROM_MS, "to": FROM_MS + 7_200_000})
    assert resp.status_code == 400
    assert "exceeds" in resp.json()["error"]
    assert source.queries == []  # refused without touching finelog


def test_oversized_result_is_a_400_telling_the_caller_what_to_do():
    client, _ = _client(FakeSource(raises=QueryResultTooLargeError("query returned 500000 rows")))
    resp = client.get("/marin/series", params={"metric": "m", "from": FROM_MS, "to": TO_MS})
    assert resp.status_code == 400
    assert "narrow the time range" in resp.json()["error"]


def test_repeated_identical_panels_hit_finelog_once():
    # A shared dashboard refreshing across N viewers must not multiply through
    # to the finelog hub; Grafana's own query caching is Enterprise-only.
    client, source = _client(FakeSource(_rows([("m", 1.0, {}, 1_784_257_200_000_000)])))
    params = {"metric": "m", "from": FROM_MS, "to": TO_MS}
    first = client.get("/marin/series", params=params)
    second = client.get("/marin/series", params=params)
    assert first.json() == second.json()
    assert len(source.queries) == 1


def test_drifting_relative_window_still_hits_the_cache():
    # Grafana sends a relative range ("now-6h to now") as absolute millis, so both
    # edges advance a little on every refresh. Keying on exact timestamps would
    # miss the cache every single time — the case the cache exists for.
    client, source = _client(FakeSource(_rows([("m", 1.0, {}, 1_784_257_200_000_000)])), cache_ttl=60.0)
    for drift_ms in (0, 1_000, 2_500):
        client.get("/marin/series", params={"metric": "m", "from": FROM_MS + drift_ms, "to": TO_MS + drift_ms})
    assert len(source.queries) == 1


def test_windows_further_apart_than_the_ttl_are_cached_separately():
    client, source = _client(FakeSource(_rows([("m", 1.0, {}, 1_784_257_200_000_000)])), cache_ttl=20.0)
    client.get("/marin/series", params={"metric": "m", "from": FROM_MS, "to": TO_MS})
    hour_later = 3_600_000
    client.get("/marin/series", params={"metric": "m", "from": FROM_MS + hour_later, "to": TO_MS + hour_later})
    assert len(source.queries) == 2


def test_differing_label_filters_are_cached_separately():
    client, source = _client(FakeSource(_rows([("m", 1.0, {"scope": "fleet"}, 1_784_257_200_000_000)])))
    base = {"metric": "m", "from": FROM_MS, "to": TO_MS}
    client.get("/marin/series", params={**base, "label.scope": "fleet"})
    client.get("/marin/series", params={**base, "label.scope": "region"})
    assert len(source.queries) == 2


def test_health_lists_configured_clusters():
    client, _ = _client(FakeSource())
    assert client.get("/health").json() == {"status": "ok", "clusters": ["marin"]}
