# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from finelog.benchmarks.grafana_dashboard_corpus import load_dashboard_corpus, render_sql


def _write_dashboard(path: Path, sql: str) -> None:
    path.write_text(
        json.dumps(
            {
                "uid": "test-dashboard",
                "title": "Test dashboard",
                "refresh": "1m",
                "panels": [
                    {
                        "title": "Nested row",
                        "panels": [
                            {
                                "title": "GPU power / watts",
                                "targets": [
                                    {
                                        "refId": "B",
                                        "url_options": {"params": [{"key": "sql", "value": sql}]},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )


def test_dashboard_corpus_renders_nested_sql_targets_with_fixed_inputs(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.json"
    _write_dashboard(
        path,
        "SELECT ${__interval_ms}, {{from}}, {{to}}, now() " "FROM telemetry_v1 WHERE cluster IN (${cluster:sqlstring})",
    )

    corpus = load_dashboard_corpus(
        path,
        start_ms=1_000,
        end_ms=5_000,
        interval_ms=250,
        clusters=("marin", "quoted'cluster"),
    )

    assert corpus.uid == "test-dashboard"
    assert corpus.title == "Test dashboard"
    assert corpus.refresh == "1m"
    assert len(corpus.queries) == 1
    query = corpus.queries[0]
    assert query.name == "gpu_power_watts_b"
    assert "SELECT 250, to_timestamp_millis(1000), to_timestamp_millis(5000)" in query.sql
    assert "to_timestamp_millis(5000) FROM" in query.sql
    assert "cluster IN ('marin','quoted''cluster')" in query.sql


def test_render_sql_rejects_macros_the_harness_does_not_model() -> None:
    with pytest.raises(ValueError, match="unresolved macro"):
        render_sql(
            "SELECT ${unsupported}",
            start_ms=1,
            end_ms=2,
            interval_ms=1,
            clusters=("marin",),
        )


def test_accelerators_dashboard_exposes_every_panel_query_to_the_benchmark() -> None:
    dashboard = Path(__file__).resolve().parents[3] / "infra/grafana/dashboards/accelerators.json"

    corpus = load_dashboard_corpus(
        dashboard,
        start_ms=1_000,
        end_ms=5_000,
        interval_ms=250,
        clusters=("marin",),
    )

    assert {query.name for query in corpus.queries} == {
        "faulted_gpus",
        "gpu_power",
        "gpu_power_by_device_model",
        "gpu_power_by_training_run",
        "gpu_temperature_distribution",
        "gpus_reporting_a_hardware_fault",
        "hbm_in_use_by_cluster",
        "hottest_gpu",
        "mean_tensor_core_activity",
        "mean_utilization",
        "nodes_reporting",
        "peak_gpu_temperature_by_cluster",
        "sm_utilization_distribution",
        "telemetry_freshness",
        "tensor_core_activity_by_cluster",
    }


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "interval_ms", "clusters", "message"),
    [
        (2, 2, 1, ("marin",), "start_ms"),
        (1, 2, 0, ("marin",), "interval_ms"),
        (1, 2, 1, (), "cluster"),
    ],
)
def test_render_sql_rejects_invalid_fixed_inputs(
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    clusters: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        render_sql(
            "SELECT 1",
            start_ms=start_ms,
            end_ms=end_ms,
            interval_ms=interval_ms,
            clusters=clusters,
        )
