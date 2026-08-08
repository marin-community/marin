# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from finelog.deploy.shadow import UNMATCHED_VALUE, ShadowReport, dashboard_variables, missing_namespace


def test_a_planning_error_names_the_namespace_the_snapshot_lacks() -> None:
    error = "Error during planning: table 'datafusion.public.iris.task_state' not found"

    assert missing_namespace(error) == "iris.task_state"


def test_an_ordinary_query_failure_is_not_read_as_a_missing_namespace() -> None:
    assert missing_namespace("Arrow error: Invalid argument error: column types must match") is None


def test_a_report_says_which_namespaces_went_unexercised() -> None:
    report = ShadowReport(
        namespaces_expected=("log", "telemetry_v1"),
        namespaces_rehydrated=("log", "telemetry_v1"),
        queries_run=25,
        queries_skipped={"jobs.json:fleet_task_states": "iris.task_state"},
        dashboards_run=("jobs.json",),
    )

    described = report.describe()

    assert report.passed()
    assert "namespaces rehydrated: 2/2" in described
    assert "dashboard queries green: 25 across jobs.json" in described
    assert "dashboard queries not run: 1" in described
    assert "iris.task_state" in described
    assert described.endswith("SHADOW PASS")


def test_a_failure_makes_the_report_fail() -> None:
    report = ShadowReport(failures=["log.json:rate: boom"])

    assert not report.passed()
    assert report.describe().endswith("SHADOW FAIL")


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
