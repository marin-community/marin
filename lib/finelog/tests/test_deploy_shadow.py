# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from finelog.client import LogClient
from finelog.deploy.shadow import (
    UNMATCHED_VALUE,
    ShadowReport,
    dashboard_variables,
    missing_namespace,
    run_dashboard_corpus,
)
from finelog.embedded import is_available, require_embedded_server

DASHBOARD_DIR = Path(__file__).resolve().parents[3] / "infra/grafana/dashboards"


@pytest.fixture
def embedded_server(tmp_path):
    if not is_available():
        pytest.skip("finelog native server extension (finelog_server) not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "log-server"))
    try:
        yield server
    finally:
        server.stop()


def test_a_planning_error_names_the_namespace_the_snapshot_lacks() -> None:
    error = "Error during planning: table 'datafusion.public.iris.task_state' not found"

    assert missing_namespace(error) == "iris.task_state"


def test_an_ordinary_query_failure_is_not_read_as_a_missing_namespace() -> None:
    assert missing_namespace("Arrow error: Invalid argument error: column types must match") is None


def test_a_dashboard_query_over_a_namespace_this_store_lacks_is_not_a_failure(embedded_server) -> None:
    # The dashboards read every namespace any Marin service writes; a store holds
    # only what its own clients registered. Those queries must be reported as not
    # run — counting them as failures would make every real snapshot fail, and
    # counting them as green would claim coverage the check never had.
    client = LogClient.connect(embedded_server.address)
    try:
        rehydrated = tuple(sorted(client.list_namespaces()))
    finally:
        client.close()
    report = ShadowReport(namespaces_expected=rehydrated, namespaces_rehydrated=rehydrated)

    run_dashboard_corpus(
        embedded_server.address,
        sorted(DASHBOARD_DIR.glob("*.json")),
        start_ms=1_000,
        end_ms=5_000,
        interval_ms=250,
        clusters=("marin",),
        report=report,
    )

    assert report.passed(), report.describe()
    assert report.queries_skipped, "the dashboards read namespaces only a live deployment registers"
    assert set(report.queries_skipped.values()).isdisjoint(rehydrated)
    if "telemetry_v1" in rehydrated:
        assert report.queries_run > 0
        assert "accelerators.json" in report.dashboards_run


def test_a_report_says_which_namespaces_went_unexercised() -> None:
    report = ShadowReport(
        namespaces_expected=("log", "telemetry_v1"),
        namespaces_rehydrated=("log", "telemetry_v1"),
        queries_run=25,
        queries_skipped={"jobs.json:fleet_task_states": "iris.task_state"},
        dashboards_run=("jobs.json",),
    )

    described = report.describe()

    # An operator reads this to decide whether a green run covered anything.
    assert "dashboard queries green: 25 across jobs.json" in described
    assert "dashboard queries not run: 1" in described
    assert "iris.task_state" in described


def test_a_dashboards_own_variables_get_values_the_snapshot_can_supply(tmp_path: Path) -> None:
    dashboard = tmp_path / "runs.json"
    dashboard.write_text(
        json.dumps(
            {
                "panels": [
                    {
                        "title": "loss",
                        "targets": [
                            {
                                "refId": "A",
                                "url_options": {
                                    "params": [
                                        {
                                            "key": "sql",
                                            "value": (
                                                "SELECT 1 WHERE cluster IN (${cluster:sqlstring}) "
                                                "AND run IN (${run:sqlstring})"
                                            ),
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )

    # `cluster` comes from the snapshot; a run id does not, so the query still
    # plans and scans but matches nothing.
    assert dashboard_variables(dashboard, ["marin", "marin-eu"]) == {
        "cluster": ["marin", "marin-eu"],
        "run": [UNMATCHED_VALUE],
    }
