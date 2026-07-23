# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for live finelog, Iris, GitHub, and W&B bridge sources."""

import json

import httpx
import pyarrow as pa
import pytest
from config import ClusterTarget
from conftest import bridge_config
from errors import UpstreamError
from finelog_health import FinelogRole
from finelog_source import FinelogSource
from github_source import GithubSource
from iris_source import IrisSource
from k8s_source import K8sFleet
from nightly_config import NIGHTLY_LANES
from server import create_app
from starlette.testclient import TestClient
from wandb_source import WandbSource

TARGET = ClusterTarget(name="marin", project="p", zone="z", instance_filter="f", controller_filter="c")


def _iris(handler) -> IrisSource:
    source = IrisSource(TARGET, timeout=5.0)
    source._base_url = "http://controller:10000"  # skip GCE discovery
    source._client = httpx.Client(transport=httpx.MockTransport(handler), headers={"content-type": "application/json"})
    return source


def _github(handler, token: str | None = None) -> GithubSource:
    source = GithubSource(token=token, timeout=5.0)
    source._client = httpx.Client(transport=httpx.MockTransport(handler), headers=source._client.headers)
    return source


def _wandb(handler) -> WandbSource:
    source = WandbSource(timeout=5.0)
    source._client = httpx.Client(transport=httpx.MockTransport(handler), headers=source._client.headers)
    return source


class _FakeLogClient:
    def __init__(self, raises: Exception | None = None) -> None:
        self._raises = raises

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        assert sql == 'SELECT * FROM "log" LIMIT 1'
        assert max_rows == 1
        if self._raises is not None:
            raise self._raises
        return pa.table({"1": [1]})


def _finelog(raises: Exception | None = None) -> FinelogSource:
    source = FinelogSource(TARGET, timeout_ms=5_000)
    source._client = _FakeLogClient(raises)
    return source


def test_finelog_health_probes_the_log_query_path():
    row = _finelog().health()
    assert isinstance(row.latency_ms, int)
    assert (row.cluster, row.server, row.role) == ("marin", "finelog-marin", FinelogRole.HUB)
    assert row.responsive is True
    assert (row.ready, row.desired, row.error_class) == (1, 1, "")


def test_finelog_health_reports_query_failures_without_raising():
    row = _finelog(TimeoutError("slow")).health()
    assert (row.cluster, row.server, row.role) == ("marin", "finelog-marin", FinelogRole.HUB)
    assert row.responsive is False
    assert (row.ready, row.desired, row.latency_ms, row.error_class) == (0, 1, None, "TimeoutError")


def test_finelog_health_does_not_mask_programming_errors():
    with pytest.raises(ValueError, match="bug"):
        _finelog(ValueError("bug")).health()


# --- IrisSource ------------------------------------------------------------


def test_jobs_splits_inflight_from_terminal_and_names_states():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/ExecuteRawQuery")
        return httpx.Response(
            200,
            json={
                "columns": [{"name": "state", "type": "integer"}, {"name": "n", "type": "integer"}],
                "rows": ["[3, 5]", "[4, 10]", "[6, 2]", "[99, 1]"],
            },
        )

    assert _iris(handler).jobs() == [
        {"bucket": "inflight", "state": "running", "count": 5},
        {"bucket": "last24h", "state": "succeeded", "count": 10},
        {"bucket": "last24h", "state": "killed", "count": 2},
        {"bucket": "last24h", "state": "state_99", "count": 1},
    ]


def test_workers_aggregates_healthy_only_per_region():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "hasMore": False,
                "workers": [
                    {
                        "healthy": True,
                        "metadata": {
                            "cpuCount": 4,
                            "memoryBytes": "100",
                            "device": {"tpu": {"count": 8}},
                            "attributes": {"region": {"stringValue": "us-east5"}},
                        },
                    },
                    {
                        "healthy": True,
                        "metadata": {"cpuCount": 2, "attributes": {"region": {"stringValue": "us-east5"}}},
                    },
                    {"healthy": False, "metadata": {"attributes": {"region": {"stringValue": "us-east5"}}}},
                    {"healthy": True, "metadata": {"cpuCount": 1}},  # no region -> unknown
                ],
            },
        )

    assert _iris(handler).workers() == [
        {"region": "unknown", "healthy": 1, "cpu_millicores": 1000, "memory_bytes": 0, "tpu_chips": 0},
        {"region": "us-east5", "healthy": 2, "cpu_millicores": 6000, "memory_bytes": 100, "tpu_chips": 8},
    ]


def test_workers_follows_pagination():
    pages = [
        {"hasMore": True, "workers": [{"healthy": True, "metadata": {"attributes": {"region": {"stringValue": "a"}}}}]},
        {
            "hasMore": False,
            "workers": [{"healthy": True, "metadata": {"attributes": {"region": {"stringValue": "b"}}}}],
        },
    ]
    seen_offsets = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_offsets.append(json.loads(request.content)["query"]["offset"])
        return httpx.Response(200, json=pages[len(seen_offsets) - 1])

    regions = {row["region"] for row in _iris(handler).workers()}
    assert regions == {"a", "b"}
    assert seen_offsets == [0, 1]


def test_health_reports_reachable_with_latency():
    result = _iris(lambda request: httpx.Response(200, json={})).health()
    assert result[0]["reachable"] is True
    assert result[0]["up"] == 1
    assert isinstance(result[0]["latency_ms"], int)


def test_health_reports_unreachable_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert _iris(handler).health() == [{"reachable": False, "up": 0, "latency_ms": None, "error": "down"}]


def test_controller_non_200_raises_upstream_error():
    with pytest.raises(UpstreamError) as excinfo:
        _iris(lambda request: httpx.Response(503)).jobs()
    assert excinfo.value.source == "iris"
    assert excinfo.value.status_code == 502


# --- GithubSource ----------------------------------------------------------


def _run(conclusion, status="completed", started="2026-07-17T03:00:00Z", updated="2026-07-17T03:05:00Z"):
    return {
        "id": 1,
        "conclusion": conclusion,
        "status": status,
        "head_sha": "abcdef1234567890",
        "run_started_at": started,
        "created_at": started,
        "updated_at": updated,
        "html_url": "https://x",
        "actor": {"login": "someone"},
    }


def test_ferries_shape_and_success_rate():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/actions/workflows/" in request.url.path
        return httpx.Response(200, json={"workflow_runs": [_run("success"), _run("failure")]})

    rows = _github(handler).ferries()
    # 5 tiers across the 3 configured ferry groups, 2 runs each.
    assert len(rows) == 10
    first = rows[0]
    assert first["sha"] == "abcdef1" and first["duration_seconds"] == 300 and first["success_rate"] == 0.5
    assert first["group"] == "Canary ferry"


def test_builds_maps_state_and_finalized_success_rate():
    nodes = [
        {
            "oid": "1",
            "abbreviatedOid": "1",
            "messageHeadline": "a",
            "committedDate": "2026-07-17T03:00:00Z",
            "url": "u",
            "author": {"user": {"login": "x"}},
            "statusCheckRollup": {"state": "SUCCESS"},
        },
        {
            "oid": "2",
            "abbreviatedOid": "2",
            "messageHeadline": "b",
            "committedDate": "2026-07-17T03:00:00Z",
            "url": "u",
            "author": {"user": {"login": "x"}},
            "statusCheckRollup": {"state": "FAILURE"},
        },
        {
            "oid": "3",
            "abbreviatedOid": "3",
            "messageHeadline": "c",
            "committedDate": "2026-07-17T03:00:00Z",
            "url": "u",
            "author": {"user": None, "name": "y"},
            "statusCheckRollup": None,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"repository": {"ref": {"target": {"history": {"nodes": nodes}}}}}})

    rows = _github(handler).builds()
    assert [r["state"] for r in rows] == ["SUCCESS", "FAILURE", "NONE"]
    assert rows[0]["success_rate"] == 0.5  # 1 success of 2 finalized (NONE excluded)
    assert rows[2]["author"] == "y"  # falls back to author.name when no user


def test_github_graphql_errors_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errors": [{"message": "bad"}]})

    with pytest.raises(UpstreamError) as excinfo:
        _github(handler).builds()
    assert excinfo.value.source == "github"


def test_wandb_points_follow_report_runset_and_drop_null_metric_rows():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "query Report" in body["query"]:
            spec = {"blocks": [{"type": "panel-grid", "metadata": {"runSets": [{"selections": {"tree": ["hero"]}}]}}]}
            return httpx.Response(
                200,
                json={"data": {"view": {"displayName": "Hero report", "spec": json.dumps(spec)}}},
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "project": {
                        "run": {
                            "state": "running",
                            "sampledHistory": [
                                [
                                    {"throughput/total_tokens": 10, "throughput/mfu": 0.42},
                                    {"throughput/total_tokens": 20, "throughput/mfu": None},
                                ]
                            ],
                        }
                    }
                }
            },
        )

    assert _wandb(handler).points("mfu") == [
        {
            "chart": "MFU (%)",
            "run": "hero",
            "run_state": "running",
            "tokens": 10,
            "value": 0.42,
            "report_title": "Hero report",
            "report_url": (
                "https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ"
            ),
        }
    ]


def test_wandb_rejects_unknown_chart_without_network():
    with pytest.raises(ValueError, match="unknown W&B chart"):
        _wandb(lambda request: pytest.fail("unexpected request")).points("nope")


# --- endpoint routing / fail-loud ------------------------------------------


class _FakeIris:
    def __init__(self, target, *, raises=None, rows=None):
        self._target = target
        self._raises = raises
        self._rows = rows or []

    @property
    def target(self):
        return self._target

    def jobs(self):
        if self._raises:
            raise self._raises
        return self._rows


def _app(iris_source, github_source: GithubSource | None = None) -> TestClient:
    github = github_source or GithubSource(token=None, timeout=5.0)
    return TestClient(
        create_app(bridge_config(), {}, {"marin": iris_source}, github, K8sFleet(()), WandbSource(timeout=5.0))
    )


def test_iris_endpoint_returns_rows():
    client = _app(_FakeIris(TARGET, rows=[{"bucket": "inflight", "state": "running", "count": 3}]))
    assert client.get("/iris/marin/jobs").json() == [{"bucket": "inflight", "state": "running", "count": 3}]


def test_dead_controller_fails_loud_not_empty():
    client = _app(_FakeIris(TARGET, raises=UpstreamError("iris", "controller unreachable", status_code=504)))
    resp = client.get("/iris/marin/jobs")
    assert resp.status_code == 504
    assert resp.json()["source"] == "iris"


def test_unknown_cluster_on_iris_route_is_400():
    assert _app(_FakeIris(TARGET)).get("/iris/nope/jobs").status_code == 400


def test_nightlies_endpoint_returns_linked_long_cells():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "workflow_runs": [
                    {
                        "id": 1,
                        "status": "completed",
                        "conclusion": "success",
                        "head_sha": "abcdef1234567890",
                        "created_at": "2026-07-17T06:05:00Z",
                        "run_started_at": "2026-07-17T06:05:00Z",
                        "updated_at": "2026-07-17T07:30:00Z",
                        "html_url": "https://x",
                        "event": "schedule",
                    }
                ]
            },
        )

    rows = _app(_FakeIris(TARGET), github_source=_github(handler)).get("/github/nightlies").json()
    assert len(rows) == 7 * len(NIGHTLY_LANES)
    lane_ids = {lane.id for lane in NIGHTLY_LANES}
    assert {row["lane_id"] for row in rows} == lane_ids
    assert all("workflow_url" in row and "lane_order" in row for row in rows)
    assert any(row["url"] == "https://x" for row in rows)
