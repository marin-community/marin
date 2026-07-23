# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests over the provisioning tree: the alerting YAML parses with resolvable
datasource UIDs and refIds, every rule's query URL answers on the bridge, and
dashboard datasources exist. These files only otherwise fail inside a deployed
Grafana, which is the most expensive place to find out."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from config import ClusterTarget
from conftest import bridge_config, healthy_k8s_routes, k8s_api, make_k8s_source
from dashboard_stitch import stitch_all
from finelog_health import FinelogHealth, FinelogRole
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


def _stitched_dashboards() -> dict[str, dict]:
    """Every dashboard as Grafana actually renders it: panelRef markers resolved.

    The checks below assert on the deployed shape, not the templated source —
    a panel's real datasource/columns/thresholds live in its fragment file once
    it's been extracted behind a panelRef.
    """
    return stitch_all(DASHBOARDS, DASHBOARDS / "panels")


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


def test_every_rule_alerts_on_nodata_and_error():
    # The alert endpoints return explicit zeros when healthy, so NoData/exec
    # errors can only mean the pipeline itself broke — which must page too.
    for rule in _rules():
        assert rule["noDataState"] == "Alerting", rule["uid"]
        assert rule["execErrState"] == "Alerting", rule["uid"]
        assert rule["labels"]["severity"] in VALID_SEVERITIES, rule["uid"]


class _FakeIris:
    def __init__(self, name: str) -> None:
        self.target = ClusterTarget(name=name, project="p", zone="z", instance_filter="f", controller_filter="c")

    def health(self) -> list[dict]:
        return [{"reachable": True, "up": 1, "latency_ms": 3}]


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
            GithubSource(token=None, timeout=5.0),
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
    for policy in _load(ALERTING / "policies.yaml")["policies"]:
        assert policy["receiver"] in contact_points
        for route in policy.get("routes", []):
            assert route["receiver"] in contact_points


def test_critical_contact_point_reaches_email_and_slack():
    points = {point["name"]: point for point in _load(ALERTING / "contact-points.yaml")["contactPoints"]}
    critical_types = {receiver["type"] for receiver in points["ops-critical"]["receivers"]}
    assert critical_types == {"email", "slack"}
    for point in points.values():
        for receiver in point["receivers"]:
            if receiver["type"] == "slack":
                assert receiver["settings"]["url"] == "$SLACK_ALERTS_WEBHOOK"


def test_finelog_health_alert_pages_critical_after_five_minutes():
    (rule,) = [rule for rule in _rules() if rule["uid"] == "finelog-fleet-unhealthy"]
    assert rule["for"] == "5m"
    assert rule["labels"]["severity"] == "critical"
    assert rule["data"][0]["datasourceUid"] == "finelog-marin"
    assert rule["data"][0]["model"]["url"] == "/alerts/fleet_health"


def test_k8s_dashboard_shows_finelog_fleet_health():
    dashboard = _stitched_dashboards()["k8s.json"]
    (panel,) = [
        panel
        for panel in dashboard["panels"]
        if any(target.get("url") == "/fleet_health" for target in panel.get("targets", []))
    ]
    assert panel["datasource"]["uid"] == "finelog-marin"
    selectors = {column["selector"] for column in panel["targets"][0]["columns"]}
    assert {"cluster", "server", "responsive", "ready", "desired", "latency_ms"} <= selectors


def test_dashboard_filter_expressions_reference_selected_columns():
    # Infinity's backend parser applies filterExpression to the frame built from
    # `columns`, so every field a filter references must also be selected.
    literals = {"true", "false", "null"}
    for name, dashboard in _stitched_dashboards().items():
        for panel in dashboard["panels"]:
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
        for panel in dashboard["panels"]:
            uid = (panel.get("datasource") or {}).get("uid")
            if uid is None or uid.startswith("${"):  # row panels / template variables
                continue
            assert uid in uids, f"{name} panel {panel.get('id')}: unknown datasource {uid!r}"


def test_stat_panels_use_grafana_reduce_options_schema():
    for name, dashboard in _stitched_dashboards().items():
        for panel in dashboard["panels"]:
            if panel.get("type") != "stat":
                continue
            reduce_options = panel.get("options", {}).get("reduceOptions", {})
            assert "calc" not in reduce_options, f"{name} panel {panel['id']}: use calcs, not calc"
            assert reduce_options.get("calcs"), f"{name} panel {panel['id']}: missing reduction"
