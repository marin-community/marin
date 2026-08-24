# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for live finelog, Iris, GitHub, and W&B bridge sources."""

import json
from datetime import UTC, datetime, timedelta

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


def _github(handler) -> GithubSource:
    source = GithubSource(auth=None, timeout=5.0)
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


def test_peers_reports_controller_heartbeat_reachability():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/ListPeers")
        return httpx.Response(
            200,
            json={
                "peers": [
                    {
                        "peerId": "cw-a",
                        "controllerAddress": "https://iris-cw-a.example",
                        "reachable": True,
                        "lastContactMs": "1",
                    },
                    {
                        "peerId": "cw-b",
                        "controllerAddress": "https://iris-cw-b.example",
                        "reachable": False,
                        "lastContactMs": "1",
                    },
                ]
            },
        )

    rows = _iris(handler).peers()
    assert [(row["peer"], row["state"], row["value"]) for row in rows] == [
        ("cw-a", "reachable", 0),
        ("cw-b", "unreachable", 1),
    ]
    assert all(row["last_contact_age_seconds"] > 0 for row in rows)


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


def _history_handler(found_in: str, points: list[dict], asked: list[tuple[str, list[str]]]):
    """Serve `points` from the project named `found_in`; record each project asked."""

    def handler(request: httpx.Request) -> httpx.Response:
        variables = json.loads(request.content)["variables"]
        asked.append((variables["project"], json.loads(variables["specs"][0])["keys"]))
        if variables["project"] != found_in:
            return httpx.Response(200, json={"data": {"project": None}})
        return httpx.Response(
            200,
            json={"data": {"project": {"run": {"state": "running", "sampledHistory": [points]}}}},
        )

    return handler


def test_wandb_run_history_searches_projects_and_drops_null_metric_rows():
    asked: list[tuple[str, list[str]]] = []
    points = [{"_step": 0, "train/loss": 3.1}, {"_step": 1, "train/loss": None}, {"_step": 2, "train/loss": 2.7}]

    rows = _wandb(_history_handler("marin", points, asked)).run_history("hero-run", metric="train/loss")

    # `_step` is the x axis because levanter logs through wandb.log(..., step=<step>).
    assert asked == [("marin_moe", ["_step", "train/loss"]), ("marin", ["_step", "train/loss"])]
    assert rows == [
        {
            "run": "hero-run",
            "project": "marin",
            "run_url": "https://wandb.ai/marin-community/marin/runs/hero-run",
            "step": step,
            "value": value,
        }
        for step, value in ((0, 3.1), (2, 2.7))
    ]


def test_wandb_run_history_pins_an_explicit_project_without_searching():
    asked: list[tuple[str, list[str]]] = []
    handler = _history_handler("marin_moe", [{"_step": 7, "train/loss": 2.5}], asked)

    rows = _wandb(handler).run_history("hero-run", metric="train/loss", project="marin_moe")

    assert [project for project, _ in asked] == ["marin_moe"]
    assert [row["step"] for row in rows] == [7]


def _activity_handler(found_in: str, run: dict, asked: list[str], tps_points: list[dict] = ()):
    """Serve `run` for the activity query and `tps_points` for the reference-rate history.

    Only the activity search is recorded in `asked`; the token-rate history read that
    follows it asks the project the run was already found in, so it carries no new
    routing information.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        variables = json.loads(request.content)["variables"]
        if "specs" in variables:  # the reference-tps history read, not the activity search
            if variables["project"] != found_in:
                return httpx.Response(200, json={"data": {"project": None}})
            history = {"state": "running", "sampledHistory": [list(tps_points)]}
            return httpx.Response(200, json={"data": {"project": {"run": history}}})
        asked.append(variables["project"])
        if variables["project"] != found_in:
            return httpx.Response(200, json={"data": {"project": None}})
        return httpx.Response(200, json={"data": {"project": {"run": run}}})

    return handler


def test_wandb_run_activity_separates_active_time_from_downtime():
    # `_runtime` is W&B's own count of the seconds a process was alive, restored at each
    # resume, so it is the run's active time across restarts and never includes the wait
    # between two attempts. Wall clock here is four days, of which ninety hours ran.
    # Progress efficiency divides the tokens this run produced by the mean rate times wall.
    # The reference rate is the mean over the history (2.5M here), not the summary's last
    # step, so a checkpoint step logging a low rate cannot skew it. This run is a fresh id
    # resumed at step 39,000 with a 10M-token batch, so total_tokens is 10M*(step+1): the
    # inherited count before its first step is reconstructed as 390.01e9 * 39000/39001, i.e.
    # 390e9, not the earliest sample's 390.01e9 (which would drop that first batch too).
    # Crediting the full 1038e9 over four days would read 120%; only 1038e9 - 390e9 counts,
    # and 648e9 / (2.5e6 * 345600) is 0.75.
    asked: list[str] = []
    run = {
        "state": "running",
        "createdAt": "2026-08-20T02:00:00Z",
        "heartbeatAt": "2026-08-24T02:00:00Z",
        "summaryMetrics": json.dumps({"_runtime": 90 * 3_600, "throughput/total_tokens": 1_038_000_000_000}),
    }
    tps_points = [
        {"_step": 39_000, "throughput/total_tokens": 390_010_000_000, "throughput/tokens_per_second": 2_000_000},
        {"_step": 78_001, "throughput/total_tokens": 780_020_000_000, "throughput/tokens_per_second": 3_000_000},
    ]

    (row,) = _wandb(_activity_handler("marin_moe", run, asked, tps_points)).run_activity("hero-run")

    assert asked == ["marin_moe"]
    assert row == {
        "run": "hero-run",
        "project": "marin_moe",
        "run_url": "https://wandb.ai/marin-community/marin_moe/runs/hero-run",
        "state": "running",
        "active_seconds": 324_000.0,
        "wall_seconds": 345_600.0,
        "downtime_seconds": 21_600.0,
        "active_share": 0.9375,
        "reference_tps": 2_500_000.0,
        "progress_efficiency": pytest.approx(0.75),
    }


def test_wandb_run_activity_credits_a_from_scratch_run_its_first_step():
    # A run started from step 0 inherited nothing, so its baseline reconstructs to zero and
    # its first step counts: total_tokens * 0 / 1 == 0. Without the reconstruction the first
    # sample would be taken as the baseline and a one-step run would report null. Here the
    # run has produced 100e9 tokens over a 100000s wall clock at a 2M reference rate, so
    # progress efficiency is 100e9 / (2e6 * 100000), i.e. 0.5.
    asked: list[str] = []
    run = {
        "state": "running",
        "createdAt": "2026-08-20T02:00:00Z",
        "heartbeatAt": "2026-08-21T05:46:40Z",
        "summaryMetrics": json.dumps({"_runtime": 90_000, "throughput/total_tokens": 100_000_000_000}),
    }
    tps_points = [{"_step": 0, "throughput/total_tokens": 100_000_000_000, "throughput/tokens_per_second": 2_000_000}]

    (row,) = _wandb(_activity_handler("marin_moe", run, asked, tps_points)).run_activity("hero-run")

    assert row["wall_seconds"] == 100_000.0
    assert row["reference_tps"] == 2_000_000.0
    assert row["progress_efficiency"] == pytest.approx(0.5)


def test_wandb_run_activity_reports_no_active_time_before_the_first_log():
    # A run that has been created but has logged nothing has no `_runtime` to read. The
    # tile then shows no data, which is true, rather than zero, which reads as a stall.
    # With no token rate and no tokens seen, progress efficiency is null for the same reason.
    asked: list[str] = []
    run = {
        "state": "running",
        "createdAt": "2026-08-20T02:00:00Z",
        "heartbeatAt": "2026-08-20T02:10:00Z",
        "summaryMetrics": "{}",
    }

    (row,) = _wandb(_activity_handler("marin", run, asked)).run_activity("hero-run")

    assert asked == ["marin_moe", "marin"]
    assert (row["active_seconds"], row["downtime_seconds"], row["active_share"]) == (None, None, None)
    assert row["wall_seconds"] == 600.0
    assert (row["reference_tps"], row["progress_efficiency"]) == (None, None)


def test_wandb_run_activity_fails_loud_when_no_project_has_the_run():
    asked: list[str] = []

    with pytest.raises(UpstreamError) as excinfo:
        _wandb(_activity_handler("nowhere", {}, asked)).run_activity("hero-run")

    assert excinfo.value.status_code == 404
    assert asked == ["marin_moe", "marin"]


def test_wandb_run_history_fails_loud_when_no_project_has_the_run():
    asked: list[tuple[str, list[str]]] = []
    handler = _history_handler("nowhere", [], asked)

    with pytest.raises(UpstreamError) as excinfo:
        _wandb(handler).run_history("hero-run", metric="train/loss")

    assert excinfo.value.source == "wandb"
    assert excinfo.value.status_code == 404
    assert [project for project, _ in asked] == ["marin_moe", "marin"]


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

    def peers(self):
        if self._raises:
            raise self._raises
        return self._rows


def _app(iris_source, github_source: GithubSource | None = None) -> TestClient:
    github = github_source or GithubSource(auth=None, timeout=5.0)
    return TestClient(
        create_app(bridge_config(), {}, {"marin": iris_source}, github, K8sFleet(()), WandbSource(timeout=5.0))
    )


def test_iris_endpoint_returns_rows():
    client = _app(_FakeIris(TARGET, rows=[{"bucket": "inflight", "state": "running", "count": 3}]))
    assert client.get("/iris/marin/jobs").json() == [{"bucket": "inflight", "state": "running", "count": 3}]


def test_iris_peers_endpoint_returns_heartbeat_rows():
    rows = [{"peer": "cw-a", "state": "unreachable", "value": 1}]
    assert _app(_FakeIris(TARGET, rows=rows)).get("/iris/marin/peers").json() == rows


def test_dead_controller_fails_loud_not_empty():
    client = _app(_FakeIris(TARGET, raises=UpstreamError("iris", "controller unreachable", status_code=504)))
    resp = client.get("/iris/marin/jobs")
    assert resp.status_code == 504
    assert resp.json()["source"] == "iris"


def test_unknown_cluster_on_iris_route_is_400():
    assert _app(_FakeIris(TARGET)).get("/iris/nope/jobs").status_code == 400


def test_nightlies_endpoint_returns_linked_long_cells():
    run_day = (datetime.now(UTC) - timedelta(days=1)).replace(hour=6, minute=5, second=0, microsecond=0)

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
                        "created_at": run_day.isoformat(),
                        "run_started_at": run_day.isoformat(),
                        "updated_at": run_day.replace(hour=7, minute=30).isoformat(),
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
