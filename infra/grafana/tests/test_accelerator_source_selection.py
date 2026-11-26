# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from threading import Lock

import duckdb
import pytest
from accelerator_observability import accelerator_overview_dataset
from dashboard_dataset import DashboardDataset, project_dataset


@pytest.fixture
def accelerator_database():
    with duckdb.connect() as database:
        database.execute(
            """CREATE TABLE "telemetry_v1.node_agent"(
                cluster VARCHAR, node_name VARCHAR, service VARCHAR, name VARCHAR,
                attributes_json VARCHAR, timestamp_ms BIGINT, value DOUBLE)"""
        )
        database.execute(
            """CREATE TABLE "levanter.metrics"(
                cluster VARCHAR, node_name VARCHAR, run_id VARCHAR, step BIGINT, timestamp_ms BIGINT)"""
        )
        database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
        database.execute(
            "CREATE MACRO named_struct(k1, v1, k2, v2, k3, v3) AS struct_pack(t := v1, value := v2, last_ms := v3)"
        )
        rows = []
        for cluster, node, gpu, timestamp, metric, value in [
            ("cw-a", "old-node", "GPU-0", 0, "gpu_power_watts", 100),
            ("cw-a", "node-a", "GPU-0", 1_000, "gpu_power_watts", 300),
            ("cw-a", "node-a", "GPU-1", 0, "gpu_power_watts", 400),
            ("cw-b", "node-b", "GPU-0", 900_000, "gpu_power_watts", 500),
            ("cw-a", "node-a", "GPU-0", 0, "gpu_utilization_percent", 20),
            ("cw-a", "node-a", "GPU-0", 1_000, "gpu_utilization_percent", 40),
            ("cw-a", "node-a", "GPU-1", 60_000, "gpu_utilization_percent", 90),
            ("cw-a", "node-a", "GPU-0", 0, "gpu_tensor_active_ratio", 0.2),
            ("cw-a", "node-a", "GPU-0", 1_000, "gpu_tensor_active_ratio", 0.4),
            ("cw-a", "node-a", "GPU-1", 60_000, "gpu_tensor_active_ratio", 0.9),
            ("cw-a", "node-a", "GPU-0", 0, "gpu_memory_used_bytes", 10),
            ("cw-a", "node-a", "GPU-0", 1_000, "gpu_memory_used_bytes", 30),
            ("cw-a", "node-a", "GPU-1", 0, "gpu_memory_used_bytes", 80),
            ("cw-a", "node-a", "GPU-0", 0, "gpu_temperature_celsius", 60),
            ("cw-a", "node-a", "GPU-0", 1_000, "gpu_temperature_celsius", 70),
            ("cw-a", "node-a", "GPU-1", 0, "gpu_temperature_celsius", 80),
            ("cw-a", "node-a", "GPU-0", 0, "gpu_sm_active_ratio", 0.5),
            ("cw-a", "node-a", "GPU-0", 0, "gpu_xid_error_code", 10),
            ("cw-a", "node-a", "GPU-0", 1_000, "gpu_xid_error_code", 0),
            ("cw-a", "node-a", "GPU-1", 0, "gpu_row_remap_failures", 2),
            ("cw-b", "node-b", "GPU-0", 0, "gpu_xid_error_code", 11),
            ("cw-a", "node-a", "GPU-0", 900_000, "hardware_inventory", 1),
            ("cw-a", "node-a", "GPU-1", 900_000, "hardware_inventory", 1),
        ]:
            attributes = json.dumps(
                {"gpu_uuid": gpu, "gpu_index": gpu[-1], "gpu_model": "H200" if gpu == "GPU-1" else "H100"}
            )
            rows.append((cluster, node, "iris-node-agent", metric, attributes, timestamp, value))
        database.executemany('INSERT INTO "telemetry_v1.node_agent" VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
        database.execute("INSERT INTO \"levanter.metrics\" VALUES ('cw-a', 'node-a', 'run-a', 1, 0)")
        yield database


def projected_rows(database: duckdb.DuckDBPyConnection, dataset: DashboardDataset):
    sources = {
        source.name: database.execute(source.sql.replace("FIRST_VALUE(", "FIRST(")).to_arrow_table()
        for source in dataset.sources
    }
    return project_dataset(dataset, sources, Lock(), lambda table: table.to_pylist(), dataset.max_result_rows)


@pytest.mark.parametrize(
    "view",
    [
        "power",
        "gpus",
        "nodes",
        "power_series",
        "utilization_series",
        "mean_utilization",
        "mean_tensor",
        "hottest",
        "tensor",
        "hbm",
        "temperature_series",
        "fault_count",
        "faulted",
        "freshness",
        "model_power",
        "sm",
        "temperature_distribution",
        "run_power",
    ],
)
def test_accelerator_panel_sources_preserve_combined_results(accelerator_database, view):
    combined = accelerator_overview_dataset(("cw-a",), 0, 1_200_000, 30_000)
    expected = [row for row in projected_rows(accelerator_database, combined) if row["section"] == view]
    panel = accelerator_overview_dataset(("cw-a",), 0, 1_200_000, 30_000, view)

    actual = [row for row in projected_rows(accelerator_database, panel) if row["section"] == view]

    # Several views order by time alone; packing may reorder equal-time series.
    assert sorted(actual, key=lambda row: json.dumps(row, sort_keys=True, default=str)) == sorted(
        expected, key=lambda row: json.dumps(row, sort_keys=True, default=str)
    )
    if view == "mean_utilization":
        assert actual == [{"section": "mean_utilization", "value": 50.0}]
    if view == "power":
        assert actual == [{"section": "power", "value": 0.7}]
    if view == "freshness":
        assert actual == [{"section": "freshness", "cluster": "cw-a", "gpus": 2, "nodes": 1, "lag_seconds": 1199.0}]


def test_accelerator_fleet_charts_preserve_device_means_and_cluster_fault_identity(accelerator_database):
    dataset = accelerator_overview_dataset(("cw-a", "cw-b"), 0, 1_200_000, 30_000, "hbm")

    rows = projected_rows(accelerator_database, dataset)

    assert [row for row in rows if row["section"] == "hbm"] == [
        {"section": "hbm", "t": 0, "series": "cw-a", "value": 100.0}
    ]
    assert [row for row in rows if row["section"] == "fault_count" and row["series"] == "XID error"] == [
        {"section": "fault_count", "t": 0, "series": "XID error", "value": 2.0}
    ]


@pytest.mark.parametrize("view", ["mean_utilization", "tensor", "freshness"])
def test_accelerator_panel_sources_preserve_empty_window_results(accelerator_database, view):
    accelerator_database.execute('DELETE FROM "telemetry_v1.node_agent"')
    dataset = accelerator_overview_dataset(("cw-a",), 0, 1_200_000, 30_000, view)

    actual = [row for row in projected_rows(accelerator_database, dataset) if row["section"] == view]

    assert actual == ([{"section": view, "value": None}] if view == "mean_utilization" else [])
