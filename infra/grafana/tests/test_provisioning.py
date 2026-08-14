# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests over the provisioning tree: the alerting YAML parses with resolvable
datasource UIDs and refIds, every rule's query URL answers on the bridge, and
dashboard datasources exist. These files only otherwise fail inside a deployed
Grafana, which is the most expensive place to find out."""

import re
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import pyarrow as pa
import yaml
from config import CLUSTERS, K8S_CLUSTERS, ClusterTarget
from conftest import bridge_config, healthy_k8s_routes, k8s_api, make_k8s_source
from dashboard_stitch import stitch_all
from finelog_health import FinelogHealth, FinelogRole
from fixture_bridge import _finelog
from github_source import GithubSource
from k8s_source import K8sFleet
from server import create_app
from starlette.testclient import TestClient
from wandb_source import WandbSource

ROOT = Path(__file__).resolve().parent.parent
ALERTING = ROOT / "provisioning" / "alerting"
DASHBOARDS = ROOT / "dashboards"

EXPRESSION_UID = "__expr__"
VALID_SEVERITIES = {"critical", "warning"}
STORAGE_ALERT_BYTES = 80 * 1024**4


def _stitched_dashboards() -> dict[str, dict]:
    """Every dashboard as Grafana actually renders it: panelRef markers resolved.

    The checks below assert on the deployed shape, not the templated source —
    a panel's real datasource/columns/thresholds live in its fragment file once
    it's been extracted behind a panelRef.
    """
    return stitch_all(DASHBOARDS, DASHBOARDS / "panels")


def _all_panels(dashboard: dict) -> list[dict]:
    """Every panel including those nested inside collapsed rows."""
    return [nested for panel in dashboard["panels"] for nested in (panel, *panel.get("panels", []))]


def _panel_sql(dashboard: dict) -> list[str]:
    """Every SQL string a dashboard sends, panels and template variables alike."""
    queries = [target for panel in _all_panels(dashboard) for target in panel.get("targets", [])]
    for variable in dashboard.get("templating", {}).get("list", []):
        query = variable.get("query")
        if isinstance(query, dict) and query.get("queryType") == "infinity":
            queries.append(query["infinityQuery"])
    return [
        param["value"]
        for query in queries
        for param in query.get("url_options", {}).get("params", [])
        if param["key"] == "sql"
    ]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _datasources() -> dict[str, str]:
    """Provisioned datasource uid -> bridge base path (from its loopback URL)."""
    uids = {}
    for path in (ROOT / "provisioning" / "datasources").glob("*.yaml"):
        for datasource in _load(path)["datasources"]:
            uids[datasource["uid"]] = urlsplit(datasource["url"]).path
    return uids


def _rules() -> list[dict]:
    return [rule for group in _load(ALERTING / "rules.yaml")["groups"] for rule in group["rules"]]


def test_alert_rules_have_resolvable_datasources_and_refids():
    datasource_uids = set(_datasources())
    for rule in _rules():
        ref_ids = [node["refId"] for node in rule["data"]]
        assert len(ref_ids) == len(set(ref_ids)), f"{rule['uid']}: duplicate refIds"
        assert rule["condition"] in ref_ids, f"{rule['uid']}: condition points at a missing refId"
        for node in rule["data"]:
            assert node["model"]["refId"] == node["refId"], f"{rule['uid']}: model refId mismatch"
            uid = node["datasourceUid"]
            assert uid == EXPRESSION_UID or uid in datasource_uids, f"{rule['uid']}: unknown datasource {uid!r}"


def test_alert_rules_define_nodata_and_error_behavior():
    # Most alert endpoints return explicit zeros when healthy. The storage rule
    # stays normal until the optional CoreWeave provider writes its first row.
    for rule in _rules():
        expected_no_data = "OK" if rule["uid"] == "coreweave-storage-capacity" else "Alerting"
        assert rule["noDataState"] == expected_no_data, rule["uid"]
        assert rule["execErrState"] == "Alerting", rule["uid"]
        assert rule["labels"]["severity"] in VALID_SEVERITIES, rule["uid"]


class _FakeIris:
    def __init__(self, name: str) -> None:
        self.target = ClusterTarget(name=name, project="p", zone="z", instance_filter="f", controller_filter="c")

    def health(self) -> list[dict]:
        return [{"reachable": True, "up": 1, "latency_ms": 3}]

    def peers(self) -> list[dict]:
        return [
            {
                "peer": "cw-a",
                "controller_address": "https://iris-cw-a.example",
                "state": "reachable",
                "last_contact_age_seconds": 3,
                "value": 0,
            }
        ]


class _FakeFinelog:
    def __init__(self, name: str) -> None:
        self.target = ClusterTarget(name=name, project="p", zone="z", instance_filter="f", controller_filter="c")

    def health(self) -> FinelogHealth:
        return FinelogHealth(
            cluster=self.target.name,
            server=f"finelog-{self.target.name}",
            role=FinelogRole.HUB,
            responsive=True,
            ready=1,
            desired=1,
            latency_ms=3,
            error_class="",
            error="",
        )

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        if '"infra.canary.metrics"' in sql:
            return pa.table({"probe": ["controller-ping"], "target": ["marin"], "value": [1]})
        if '"cost.events"' in sql:
            return pa.table(
                {
                    "bucket": ["marin-us-east-02a"],
                    "region": ["US-EAST-02"],
                    "value": [81 * 1024**4],
                }
            )
        return pa.table({})


def test_every_rule_query_url_answers_on_the_bridge():
    """Join each rule's datasource base path with its query URL and GET it for real."""
    iris_sources = {name: _FakeIris(name) for name in ("marin", "marin-dev")}
    finelog_sources = {"marin": _FakeFinelog("marin")}
    fleet = K8sFleet([make_k8s_source(k8s_api(healthy_k8s_routes()))])
    client = TestClient(
        create_app(
            bridge_config(),
            finelog_sources,
            iris_sources,
            GithubSource(auth=None, timeout=5.0),
            fleet,
            WandbSource(timeout=5.0),
        )
    )
    base_paths = _datasources()
    for rule in _rules():
        for node in rule["data"]:
            if node["datasourceUid"] == EXPRESSION_UID:
                continue
            model = node["model"]
            params = {p["key"]: p["value"] for p in model.get("url_options", {}).get("params", [])}
            url = base_paths[node["datasourceUid"]] + model["url"]
            response = client.get(url, params=params)
            assert response.status_code == 200, f"{rule['uid']}: GET {url} -> {response.status_code}"
            assert response.json(), f"{rule['uid']}: GET {url} returned no rows"


def test_alert_queries_select_exactly_one_numeric_column():
    # Grafana's table-alert contract: string columns become labels; the single
    # numeric column is what the threshold expression evaluates.
    for rule in _rules():
        for node in rule["data"]:
            if node["datasourceUid"] == EXPRESSION_UID:
                continue
            numeric = [c for c in node["model"]["columns"] if c["type"] == "number"]
            assert len(numeric) == 1, f"{rule['uid']}: expected exactly one numeric column"


def test_policies_reference_provisioned_contact_points():
    contact_points = {point["name"] for point in _load(ALERTING / "contact-points.yaml")["contactPoints"]}
    mute_timings = {timing["name"] for timing in _load(ALERTING / "mute-timings.yaml")["muteTimes"]}
    for policy in _load(ALERTING / "policies.yaml")["policies"]:
        assert policy["receiver"] in contact_points
        for route in policy.get("routes", []):
            assert route["receiver"] in contact_points
            assert set(route.get("mute_time_intervals", ())) <= mute_timings


def test_warning_alerts_remain_visible_without_notifications():
    (policy,) = _load(ALERTING / "policies.yaml")["policies"]
    routes_by_severity = {
        route["object_matchers"][0][2]: route
        for route in policy["routes"]
        if route["object_matchers"][0][0] == "severity"
    }

    assert routes_by_severity["critical"].get("mute_time_intervals") is None
    assert routes_by_severity["warning"]["mute_time_intervals"] == ["dashboard-only"]
    (dashboard_only,) = _load(ALERTING / "mute-timings.yaml")["muteTimes"]
    assert dashboard_only["time_intervals"] == [
        {
            "times": [{"start_time": "00:00", "end_time": "24:00"}],
            "location": "UTC",
        }
    ]


def test_coreweave_storage_alert_reads_latest_finelog_gauge_and_notifies_slack():
    (rule,) = [rule for rule in _rules() if rule["uid"] == "coreweave-storage-capacity"]
    source, threshold = rule["data"]
    sql = next(param["value"] for param in source["model"]["url_options"]["params"] if param["key"] == "sql")

    assert rule["for"] == "5m"
    assert rule["noDataState"] == "OK"
    assert rule["labels"] == {"severity": "warning", "notification": "slack"}
    assert source["datasourceUid"] == "finelog-marin"
    assert 'FROM "cost.events"' in sql
    assert "PARTITION BY provider, category, region, detail" in sql
    assert "ORDER BY usage_date DESC, seq DESC" in sql
    assert "collected_ts >= CURRENT_TIMESTAMP - INTERVAL '36 hours'" in sql
    assert {column["selector"] for column in source["model"]["columns"]} == {"bucket", "region", "value"}
    assert threshold["model"]["conditions"][0]["evaluator"] == {"type": "gt", "params": [STORAGE_ALERT_BYTES]}

    (policy,) = _load(ALERTING / "policies.yaml")["policies"]
    routes = policy["routes"]
    route = next(route for route in routes if route["object_matchers"] == [["notification", "=", "slack"]])
    muted_warning = next(route for route in routes if route["object_matchers"] == [["severity", "=", "warning"]])
    assert route["receiver"] == "ops-slack"
    assert "mute_time_intervals" not in route
    assert routes.index(route) < routes.index(muted_warning)


def test_every_slack_alert_goes_through_the_bridge_and_none_through_grafana():
    """The bridge posts every Slack alert, not Grafana. For critical alerts that is
    load-bearing — an incoming webhook never reveals the message ts that Loom needs
    to route the thread — and routing the fallback the same way keeps one channel,
    one credential, and one rendering."""
    points = {point["name"]: point for point in _load(ALERTING / "contact-points.yaml")["contactPoints"]}
    assert {receiver["type"] for receiver in points["ops-critical"]["receivers"]} == {"email", "webhook"}
    assert {receiver["type"] for receiver in points["ops-slack"]["receivers"]} == {"webhook"}
    slack_receivers = [r for point in points.values() for r in point["receivers"] if r["type"] == "slack"]
    assert slack_receivers == [], "a Slack receiver would post a second message Loom cannot route"

    (loom,) = [receiver for receiver in points["ops-critical"]["receivers"] if receiver["type"] == "webhook"]
    assert loom["settings"] == {"url": "http://127.0.0.1:8081/alerts/loom", "httpMethod": "POST"}
    (fallback,) = points["ops-slack"]["receivers"]
    assert fallback["settings"] == {"url": "http://127.0.0.1:8081/alerts/slack", "httpMethod": "POST"}


def test_finelog_health_alert_pages_critical_after_five_minutes():
    (rule,) = [rule for rule in _rules() if rule["uid"] == "finelog-fleet-unhealthy"]
    assert rule["for"] == "5m"
    assert rule["labels"]["severity"] == "critical"
    assert rule["data"][0]["datasourceUid"] == "finelog-marin"
    assert rule["data"][0]["model"]["url"] == "/alerts/fleet_health"


def test_node_deadlock_alert_pages_critical_after_five_minutes():
    (rule,) = [rule for rule in _rules() if rule["uid"] == "k8s-node-kernel-deadlock"]
    assert rule["for"] == "5m"
    assert rule["labels"]["severity"] == "critical"
    assert rule["data"][0]["model"]["url"] == "/alerts/node_deadlocks"


def test_zephyr_stall_alert_is_a_warning_after_five_minutes():
    (rule,) = [rule for rule in _rules() if rule["uid"] == "zephyr-pipeline-progress-stalled"]
    assert rule["for"] == "5m"
    assert rule["labels"]["severity"] == "warning"
    assert rule["data"][0]["model"]["url"] == "/alerts/zephyr_stalls"


def test_clusters_dashboard_shows_finelog_fleet_health():
    (panel,) = [
        panel
        for panel in _all_panels(_stitched_dashboards()["clusters.json"])
        if any(target.get("url") == "/fleet_health" for target in panel.get("targets", []))
    ]
    assert panel["datasource"]["uid"] == "finelog-marin"
    selectors = {column["selector"] for column in panel["targets"][0]["columns"]}
    assert {"cluster", "server", "responsive", "ready", "desired", "latency_ms"} <= selectors


def test_clusters_dashboard_shows_node_deadlock_and_reboot_state():
    (target,) = [
        target
        for panel in _all_panels(_stitched_dashboards()["clusters.json"])
        for target in panel.get("targets", [])
        if target.get("url") == "/nodes"
    ]
    selectors = {column["selector"] for column in target["columns"]}
    assert {
        "cluster",
        "node",
        "ready",
        "unschedulable",
        "kernel_deadlock",
        "deadlock_reason",
        "cordon_reason",
        "pending_phase",
    } <= selectors


def test_node_details_dashboard_combines_live_state_and_hardware_history():
    dashboard = _stitched_dashboards()["nodes.json"]
    panels = _all_panels(dashboard)
    state_target = next(
        target for panel in panels for target in panel.get("targets", []) if target.get("url") == "/nodes"
    )
    selectors = {column["selector"] for column in state_target["columns"]}
    assert {
        "cluster",
        "node",
        "node_pool",
        "instance_type",
        "gpu_model",
        "gpu_capacity",
        "ready",
        "unschedulable",
        "rack_name",
        "rack_slot",
        "ib_fabric",
        "ib_speed",
    } <= selectors

    sql = "\n".join(_panel_sql(dashboard))
    assert "gpu_sm_active_ratio" in sql
    assert "gpu_memory_total_bytes" in sql
    assert "gpu_nvlink_receive_bytes_per_second" in sql
    assert "gpu_pcie_transmit_bytes_per_second" in sql
    assert "node_cpu_utilization_percent" in sql
    assert "node_network_receive_bytes" in sql
    assert "json_get(attributes_json, 'node_name') IN (${node:sqlstring})" in sql


def test_node_pools_dashboard_reads_live_node_pool_state():
    dashboard = _stitched_dashboards()["node_pools.json"]
    targets = [target for panel in _all_panels(dashboard) for target in panel.get("targets", [])]

    assert targets
    assert {target["url"] for target in targets} == {"/node_pools"}
    table_target = next(target for target in targets if len(target["columns"]) > 1)
    selectors = {column["selector"] for column in table_target["columns"]}
    assert {
        "cluster",
        "node_pool",
        "instance_type",
        "compute_class",
        "current_nodes",
        "target_nodes",
        "missing_nodes",
        "in_progress_nodes",
        "queued_nodes",
        "at_target",
        "capacity_available",
        "under_quota",
        "problems",
    } <= selectors


def test_accelerators_dashboard_shows_sm_and_temperature_distributions():
    dashboard = _stitched_dashboards()["accelerators.json"]
    heatmaps = {panel["title"]: panel for panel in _all_panels(dashboard) if panel.get("type") == "heatmap"}

    assert set(heatmaps) == {"SM utilization distribution", "GPU temperature distribution"}
    assert all(panel["options"]["calculate"] for panel in heatmaps.values())
    sm_sql = _panel_sql({**dashboard, "panels": [heatmaps["SM utilization distribution"]]})
    temperature_sql = _panel_sql({**dashboard, "panels": [heatmaps["GPU temperature distribution"]]})
    assert len(sm_sql) == len(temperature_sql) == 1
    assert "name = 'gpu_sm_active_ratio'" in sm_sql[0]
    assert "name = 'gpu_temperature_celsius'" in temperature_sql[0]


def test_storage_dashboard_shows_latest_coreweave_bucket_bytes():
    dashboard = _stitched_dashboards()["storage.json"]
    (panel,) = _all_panels(dashboard)
    (sql,) = _panel_sql(dashboard)

    assert panel["type"] == "timeseries"
    assert panel["fieldConfig"]["defaults"]["unit"] == "bytes"
    assert panel["datasource"]["uid"] == "finelog-marin"
    assert 'FROM "cost.events"' in sql
    assert "provider = 'coreweave'" in sql
    assert "category = 'storage'" in sql
    assert "usage_unit = 'bytes'" in sql
    assert "PARTITION BY usage_date, provider, category, region, detail ORDER BY seq DESC" in sql
    for source in ("home.json", "infra.json"):
        (link,) = [link for link in _stitched_dashboards()[source]["links"] if link["url"] == "/d/marin-storage"]
        assert not link["keepTime"]


def test_clusters_dashboard_shows_finelog_pods_storage_and_events():
    targets = [
        target for panel in _all_panels(_stitched_dashboards()["clusters.json"]) for target in panel.get("targets", [])
    ]

    assert {("/fleet_health", "backend"), ("/finelog", "backend"), ("/finelog_events", "backend")} <= {
        (target["url"], target.get("parser")) for target in targets
    }
    pod_target = next(target for target in targets if target["url"] == "/finelog")
    selectors = {column["selector"] for column in pod_target["columns"]}
    assert {
        "cluster",
        "namespace",
        "pod",
        "node",
        "restarts",
        "cpu_request",
        "cpu_limit",
        "memory_request",
        "memory_limit",
        "startup_probe",
        "pvc",
        "storage_class",
        "storage_capacity",
    } <= selectors


def test_dashboard_filter_expressions_reference_selected_columns():
    # Infinity's backend parser applies filterExpression to the frame built from
    # `columns`, so every field a filter references must also be selected.
    literals = {"true", "false", "null"}
    for name, dashboard in _stitched_dashboards().items():
        for panel in _all_panels(dashboard):
            for target in panel.get("targets", []):
                expression = target.get("filterExpression")
                if not expression:
                    continue
                columns = target.get("columns", [])
                selected = {c["text"] for c in columns} | {c["selector"] for c in columns}
                fields = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.sub(r"'[^']*'", "", expression))) - literals
                missing = fields - selected
                assert not missing, f"{name} panel {panel.get('id')}: filter references unselected {missing}"


def test_dashboard_datasource_uids_are_provisioned():
    uids = set(_datasources())
    for name, dashboard in _stitched_dashboards().items():
        for panel in _all_panels(dashboard):
            uid = (panel.get("datasource") or {}).get("uid")
            if uid is None or uid.startswith("${"):  # row panels / template variables
                continue
            assert uid in uids, f"{name} panel {panel.get('id')}: unknown datasource {uid!r}"


def test_status_page_has_each_required_source():
    dashboard = _stitched_dashboards()["infra.json"]
    (panel,) = dashboard["panels"]
    targets = {target["refId"]: target for target in panel["targets"]}

    assert panel["datasource"]["uid"] == "status"
    assert {ref_id: target["url"] for ref_id, target in targets.items()} == {
        "N": "/github/nightlies",
        "G": "/github/builds",
        "W": "/iris/marin/workers",
        "P": "/overview/provisioning",
        "H": "/finelog/marin/query",
        "R": "/finelog/marin/query",
        "T": "/wandb/train-loss",
        "L": "/wandb/paloma-macro-loss",
        "M": "/wandb/mfu",
    }


def test_status_page_queries_provisioning_snapshot_and_region_history():
    dashboard = _stitched_dashboards()["infra.json"]
    (panel,) = dashboard["panels"]
    targets = {target["refId"]: target for target in panel["targets"]}

    assert panel["type"] == "marin-infra-panel"
    assert panel["options"]["view"] == "status"
    assert targets["P"]["url"] == "/overview/provisioning"
    snapshot_columns = {column["selector"] for column in targets["P"]["columns"]}
    assert {
        "scope",
        "ready",
        "stockout",
        "error",
        "preempted",
        "outcomes",
        "success_ratio",
        "pools_placing",
        "pools_no_ready_outcome",
    } <= snapshot_columns

    target = targets["R"]
    columns = {column["selector"]: column for column in target["columns"]}
    assert columns["series"] == {"selector": "series", "text": "region", "type": "string"}
    assert columns["value"]["type"] == "number"

    sql = next(param["value"] for param in target["url_options"]["params"] if param["key"] == "sql")
    assert "metric = 'provision_success_ratio'" in sql
    assert "metric IN ('provision_ready', 'provision_outcomes')" in sql
    assert "regexp_matches(json_get(labels, 'zone'), '^[a-z]+-[a-z]+[0-9]+-[a-z]$')" in sql
    assert "ELSE json_get(labels, 'zone') END AS series" in sql
    assert "ready / NULLIF(outcomes, 0)" in sql


def test_provisioning_render_fixture_routes_metric_query_to_region_series():
    dashboard = _stitched_dashboards()["infra.json"]
    (panel,) = dashboard["panels"]
    (target,) = [target for target in panel["targets"] if target["refId"] == "R"]
    sql = next(param["value"] for param in target["url_options"]["params"] if param["key"] == "sql")

    rows = _finelog(urlencode({"sql": sql}))

    assert rows
    assert {row["series"] for row in rows} == {
        "fleet",
        "europe-west4",
        "us-central1",
        "us-east1",
        "us-east5",
        "us-west4",
    }
    assert all({"t", "series", "value"} <= row.keys() for row in rows)


def test_stat_panels_use_grafana_reduce_options_schema():
    for name, dashboard in _stitched_dashboards().items():
        for panel in _all_panels(dashboard):
            if panel.get("type") != "stat":
                continue
            reduce_options = panel.get("options", {}).get("reduceOptions", {})
            assert "calc" not in reduce_options, f"{name} panel {panel['id']}: use calcs, not calc"
            assert reduce_options.get("calcs"), f"{name} panel {panel['id']}: missing reduction"


def test_telemetry_queries_bound_their_window_with_foldable_macros():
    # telemetry_v1 is sorted on timestamp_ms alone, so the boundary has to fold to a
    # constant for segment pruning to happen at all. `timestamp_ms >= {{from}}` compares
    # a bigint against a timestamp and turns every panel into a full namespace scan.
    for name, dashboard in _stitched_dashboards().items():
        for sql in _panel_sql(dashboard):
            if '"telemetry_v1"' not in sql:
                continue
            assert "timestamp_ms >= CAST(EXTRACT(EPOCH FROM" in sql, name
            assert "timestamp_ms < CAST(EXTRACT(EPOCH FROM" in sql, name
            assert "timestamp_ms >= {{from}}" not in sql, name


def test_cluster_column_is_only_referenced_quoted_or_as_an_alias():
    # finelog's SQL parser rejects a bare `cluster` identifier anywhere but the first
    # select-list position -- `SELECT ts AS t, cluster FROM ...` fails to parse. Quoting
    # it sidesteps the keyword entirely.
    bare = re.compile(r'(?<![\w"])cluster(?![\w"])')
    for name, dashboard in _stitched_dashboards().items():
        for sql in _panel_sql(dashboard):
            interpolated = re.sub(r"\$\{[^}]*\}", "?", sql)
            for match in bare.finditer(interpolated):
                preceding = interpolated[: match.start()].rstrip()
                assert preceding.endswith(" AS"), f"{name}: unquoted `cluster` in {sql[:160]!r}"


def test_every_sql_selector_has_a_matching_variable():
    # A selector interpolated into SQL but never declared reaches finelog as literal
    # `${name:sqlstring}` and the panel fails to parse. Shared fragments make this easy
    # to hit: a dashboard adopting one inherits its filters.
    for name, dashboard in _stitched_dashboards().items():
        variables = {variable["name"] for variable in dashboard.get("templating", {}).get("list", [])}
        for sql in _panel_sql(dashboard):
            for variable in re.findall(r"\$\{(\w+):sqlstring\}", sql):
                assert variable in variables, f"{name}: ${variable} used but not declared"


def test_dashboard_links_point_at_provisioned_dashboards():
    # Deleting a dashboard silently strands every nav link that named it.
    uids = {dashboard["uid"] for dashboard in _stitched_dashboards().values()}
    for name, dashboard in _stitched_dashboards().items():
        for link in dashboard.get("links", []):
            url = link["url"]
            if not url.startswith("/d/"):
                continue
            assert url.removeprefix("/d/").split("/")[0] in uids, f"{name}: dead link {url}"


def test_cluster_variable_lists_every_configured_cluster():
    # A cluster missing from the dropdown is invisible on every dashboard, so the list
    # tracks the clusters the bridge actually serves.
    expected = {cluster.name for cluster in CLUSTERS} | {cluster.name for cluster in K8S_CLUSTERS}
    for name, dashboard in _stitched_dashboards().items():
        variables = {v["name"]: v for v in dashboard.get("templating", {}).get("list", [])}
        if "cluster" not in variables:
            continue
        assert set(variables["cluster"]["query"].split(",")) == expected, name


def test_cluster_series_keep_one_colour_across_dashboards():
    # Colour follows the entity, not its rank: a cluster filtered out of one panel must
    # not repaint the survivors, and the same cluster reads the same on every dashboard.
    colours: dict[str, str] = {}
    for name, dashboard in _stitched_dashboards().items():
        for panel in _all_panels(dashboard):
            for override in panel.get("fieldConfig", {}).get("overrides", []):
                series = override["matcher"].get("options")
                if series not in {cluster.name for cluster in CLUSTERS} | {c.name for c in K8S_CLUSTERS}:
                    continue
                (colour,) = [p["value"]["fixedColor"] for p in override["properties"] if p["id"] == "color"]
                assert colours.setdefault(series, colour) == colour, f"{name}: {series} changes colour"
    assert len(colours) >= len(CLUSTERS)
