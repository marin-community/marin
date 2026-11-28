# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster Occupancy checked-in app against an embedded Finelog server.

The fixture scenario in ``cluster_occupancy_fixtures`` mirrors the Iris
controller namespaces with synthetic users; the window under test is
``[T0, T0 + 1h)`` with ``T0 = 2026-11-23 06:00 UTC``.
"""

import shutil
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from finelog.client import LogClient
from finelog.embedded import require_embedded_server
from marina.db import UrlDatabase
from marina.preview import install_preview_sources
from marina.server import MarinaConfig, create_app
from marina.testing import SourceServices, isolated_database

from infra.marina.apps.cluster_occupancy.backend import app as backend
from infra.marina.apps.cluster_occupancy.backend import live
from infra.marina.tests import cluster_occupancy_fixtures as fx

APP = Path(__file__).resolve().parents[1]
APPLET_ID = tomllib.loads((APP / "applet.toml").read_text())["applet_id"]
HOUR_MS = 60 * 60 * 1000
LIVE_ROWS = fx.live_rows()


@pytest.fixture(scope="module")
def finelog(tmp_path_factory):
    server = require_embedded_server()(log_dir=str(tmp_path_factory.mktemp("finelog")))
    url = f"http://127.0.0.1:{server.port}"
    client = LogClient.connect(url)
    fx.seed_occupancy_fixtures(client)
    yield url
    client.close()
    server.stop()


@pytest.fixture()
def clean_env(finelog, monkeypatch):
    """Hub credentials without fixture seams, so a test can inject its own."""
    monkeypatch.setenv("OCCUPANCY_FINELOG_URL", finelog)
    monkeypatch.delenv("OCCUPANCY_IAP_CLUSTER", raising=False)
    monkeypatch.delenv(live.LIVE_ROWS_FILE_ENV, raising=False)


@pytest.fixture()
def api(clean_env, monkeypatch):
    monkeypatch.setattr(live, "running_tasks", lambda cluster: LIVE_ROWS.get(cluster, []))
    app = backend.create_api(SourceServices(name="cluster_occupancy", data_url="file:///tmp/occupancy", database=None))
    return TestClient(app.app)


def _occupancy(api: TestClient, start_ms: int) -> dict:
    response = api.get(f"/occupancy?start_ms={start_ms}")
    assert response.status_code == 200, response.text
    return response.json()


def _cluster(body: dict, name: str) -> dict:
    return next(cluster for cluster in body["clusters"] if cluster["cluster"] == name)


def _holders(cluster: dict, node_name: str) -> list[dict]:
    node = next(node for node in cluster["nodes"] if node["node_name"] == node_name)
    return node["holders"]


def test_window_map_assigns_holders_capacity_and_racks(api):
    body = _occupancy(api, fx.WINDOW_START_MS)

    cluster_a = _cluster(body, fx.CLUSTER_A)
    assert cluster_a["state"] == "mirrored"
    assert [(node["node_name"], node["rack"]) for node in cluster_a["nodes"]] == [
        ("n1", fx.RACK_122),
        ("n2", fx.RACK_122),
        ("n3", fx.RACK_122),
        ("n5", fx.RACK_124),
        ("n6", ""),
        ("n9", ""),
    ], "rack groups sort first, rackless nodes last"
    assert {node["node_name"]: node["capacity_gpus"] for node in cluster_a["nodes"]} == {
        "n1": 8,
        "n2": 4,
        "n3": 4,
        "n5": 4,
        "n6": 8,
        "n9": 1,
    }

    n1 = _holders(cluster_a, "n1")
    assert [(holder["user_id"], holder["gpus"]) for holder in n1] == [(fx.ALICE, 4), (fx.BOB, 4)]
    assert n1[0]["finished_ms"] is None, "alice's worker is still running"
    assert n1[1]["finished_ms"] == int(fx._at(20).timestamp() * 1000), "bob's eval ends 20 minutes in"
    assert [(holder["user_id"], holder["gpus"]) for holder in _holders(cluster_a, "n2")] == [(fx.ALICE, 4)]
    assert [(holder["user_id"], holder["gpus"]) for holder in _holders(cluster_a, "n3")] == [(fx.BOB, 4)]
    assert _holders(cluster_a, "n5") == [], "job-6 finished exactly at the window start: it held nothing inside"
    assert _holders(cluster_a, "n6") == [], "job-7 starts exactly at the window end: outside the window"
    assert _holders(cluster_a, "n9") == [], "carol's old job ended and her CPU job holds no GPUs"

    cluster_b = _cluster(body, fx.CLUSTER_B)
    assert [(holder["user_id"], holder["gpus"]) for holder in _holders(cluster_b, "g1")] == [
        (fx.BOB, 8),
        (fx.CAROL, 8),
    ]
    g1 = next(node for node in cluster_b["nodes"] if node["node_name"] == "g1")
    assert g1["capacity_gpus"] == 16 and g1["gpu_type"] == "H100"

    unmirrored = _cluster(body, fx.CLUSTER_UNMIRRORED)
    assert unmirrored["state"] == "no_data" and unmirrored["reason"] == "not_mirrored" and unmirrored["nodes"] == []


def test_legend_sums_gpu_hours_per_user(api):
    body = _occupancy(api, fx.WINDOW_START_MS)
    assert body["legend"] == [
        {"user_id": fx.ALICE, "gpu_hours": 16.0},
        {"user_id": fx.BOB, "gpu_hours": 10.67},
        {"user_id": fx.CAROL, "gpu_hours": 1.33},
    ], "16 held GPUs for the full hour; bob 8 for an hour plus 8 for 20 minutes; carol 8 for 10 minutes"


def test_root_name_resolves_outside_the_window(api):
    body = _occupancy(api, fx.WINDOW_START_MS)
    cluster_a = _cluster(body, fx.CLUSTER_A)
    bob_on_n3 = _holders(cluster_a, "n3")[0]
    assert bob_on_n3["root_name"] == "/bob/eval-coordinator", "the root finished two hours before the window"
    assert bob_on_n3["priority_band"] == "PRIORITY_BAND_BATCH"
    bob_on_g1 = _holders(_cluster(body, fx.CLUSTER_B), "g1")[0]
    assert bob_on_g1["root_name"] == "/bob/eval-east-coordinator"
    carol_on_g1 = _holders(_cluster(body, fx.CLUSTER_B), "g1")[1]
    assert carol_on_g1["root_name"] == "/carol/prod-east", "a self-rooted job shows its own name"


def test_no_data_states_for_unmirrored_cluster_and_periods(api):
    body = _occupancy(api, fx.WINDOW_START_MS)
    assert _cluster(body, fx.CLUSTER_UNMIRRORED)["state"] == "no_data"

    before = _occupancy(api, fx.WINDOW_START_MS - 2 * HOUR_MS)
    cluster_b = _cluster(before, fx.CLUSTER_B)
    assert cluster_b["state"] == "no_data" and cluster_b["reason"] == "before_coverage"
    assert cluster_b["first_ms"] == int(fx._at(-40).timestamp() * 1000)
    assert _cluster(before, fx.CLUSTER_A)["state"] == "mirrored", "08a was already mirrored an hour earlier"

    boundary = _occupancy(api, fx.WINDOW_START_MS - HOUR_MS)
    assert _cluster(boundary, fx.CLUSTER_B)["state"] == "mirrored", "the window overlaps the first mirror write"

    after = _occupancy(api, fx.WINDOW_START_MS + 2 * HOUR_MS)
    for cluster in (_cluster(after, fx.CLUSTER_A), _cluster(after, fx.CLUSTER_B)):
        assert cluster["state"] == "no_data" and cluster["reason"] == "after_coverage"

    fresh = _occupancy(api, fx.WINDOW_START_MS)
    cluster_a = _cluster(fresh, fx.CLUSTER_A)
    assert cluster_a["state"] == "mirrored", "a mirror five minutes behind the window start is still fresh enough"


def test_health_lists_mirror_namespaces(api, finelog):
    body = api.get("/health").json()
    assert body["finelog"].startswith(finelog)
    assert set(body["namespaces_present"]) == {fx.JOBS, fx.JOB_CONFIG, fx.TASK_ATTEMPTS}


def test_coverage_bounds_the_slider_to_the_mirror(api):
    body = api.get("/coverage").json()
    first_a = int(fx._at(-180).timestamp() * 1000)
    last_a = int(fx._at(60).timestamp() * 1000)
    last_b = int(fx._at(-10).timestamp() * 1000)
    assert body["earliest_ms"] == first_a, "the earliest mirror write, floored to the step"
    assert body["latest_ms"] == max(last_a, last_b)
    assert body["now_ms"] >= body["latest_ms"], "the server clock is at or past the newest mirror write"
    by_cluster = {cluster["cluster"]: cluster for cluster in body["clusters"]}
    assert by_cluster[fx.CLUSTER_A]["first_ms"] == first_a
    assert by_cluster[fx.CLUSTER_UNMIRRORED]["first_ms"] is None


def test_now_view_reads_every_cluster_live(api):
    body = api.get("/now").json()
    assert body["as_of_ms"] > 0

    by_cluster = {cluster["cluster"]: cluster for cluster in body["clusters"]}
    assert by_cluster[fx.CLUSTER_UNMIRRORED]["state"] == "live", "rno2a reads live even without a mirror"
    r1 = next(node for node in by_cluster[fx.CLUSTER_UNMIRRORED]["nodes"] if node["node_name"] == "r1")
    assert r1["capacity_gpus"] == 8
    assert r1["holders"][0]["user_id"] == "dave"
    assert r1["holders"][0]["root_name"] == "/dave/train"
    assert r1["holders"][0]["finished_ms"] is None, "the live view is a snapshot: everything is running"

    # alice's two tasks on n1 aggregate into one holder with both GPUs summed.
    n1 = next(node for node in by_cluster[fx.CLUSTER_A]["nodes"] if node["node_name"] == "n1")
    assert len(n1["holders"]) == 1 and n1["holders"][0]["gpus"] == 8
    assert n1["rack"] == fx.RACK_122, "the live read still groups nodes by NVLink domain"

    # A CPU task holds no part of the GPU map.
    assert all(node["node_name"] != "n3" for node in by_cluster[fx.CLUSTER_A]["nodes"])

    assert body["legend"] == [
        {"user_id": fx.ALICE, "gpus": 8},
        {"user_id": fx.CAROL, "gpus": 8},
        {"user_id": "dave", "gpus": 8},
        {"user_id": fx.BOB, "gpus": 4},
    ], "legend merges the live clusters and drops users holding no GPUs"


def test_now_view_marks_an_unreachable_controller(clean_env, monkeypatch):
    def failing(cluster):
        if cluster == fx.CLUSTER_UNMIRRORED:
            raise ConnectionError("controller unreachable")
        return LIVE_ROWS.get(cluster, [])

    monkeypatch.setattr(live, "running_tasks", failing)
    app = backend.create_api(SourceServices(name="cluster_occupancy", data_url="file:///tmp/occupancy", database=None))
    body = TestClient(app.app).get("/now").json()
    by_cluster = {cluster["cluster"]: cluster for cluster in body["clusters"]}
    rno2a = by_cluster[fx.CLUSTER_UNMIRRORED]
    assert rno2a["state"] == "unavailable" and rno2a["reason"] == "controller_error" and rno2a["nodes"] == []
    assert by_cluster[fx.CLUSTER_A]["state"] == "live", "one unreachable controller does not blank the others"


def test_empty_hub_reports_every_cluster_as_no_data(clean_env, monkeypatch, tmp_path_factory):
    server = require_embedded_server()(log_dir=str(tmp_path_factory.mktemp("finelog-empty")))
    try:
        monkeypatch.setenv("OCCUPANCY_FINELOG_URL", f"http://127.0.0.1:{server.port}")
        app = backend.create_api(
            SourceServices(name="cluster_occupancy", data_url="file:///tmp/occupancy", database=None)
        )
        api = TestClient(app.app)
        body = _occupancy(api, fx.WINDOW_START_MS)
        assert [cluster["state"] for cluster in body["clusters"]] == ["no_data"] * 3
        assert all(cluster["reason"] == "not_mirrored" for cluster in body["clusters"])
        assert body["legend"] == []
        assert api.get("/health").json()["namespaces_present"] == []
    finally:
        server.stop()


def test_marina_mount_serves_occupancy_api(finelog, clean_env, monkeypatch, tmp_path, database_url):
    monkeypatch.setattr(live, "running_tasks", lambda cluster: LIVE_ROWS.get(cluster, []))
    apps_dir = tmp_path / "apps"
    apps_dir.mkdir()
    source = apps_dir / "cluster_occupancy"
    shutil.copytree(APP, source, ignore=shutil.ignore_patterns("web", "dist", "tests", "node_modules", "__pycache__"))
    (source / "dist").mkdir()
    (source / "dist/index.html").write_text("<h1>Cluster Occupancy</h1>")
    with isolated_database(database_url) as local_url:
        database = UrlDatabase(local_url)
        install_preview_sources(apps_dir, database)
        config = MarinaConfig(data_root=str(tmp_path / "data"), iap_audience=None, database=database)
        with TestClient(create_app(config), client=("127.0.0.1", 40000)) as client:
            body = client.get(f"/a/{APPLET_ID}/api/health").json()
            assert set(body["namespaces_present"]) == {fx.JOBS, fx.JOB_CONFIG, fx.TASK_ATTEMPTS}
