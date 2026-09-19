# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as Record

import duckdb
from accelerator_observability import accelerator_overview_dataset
from config import ClusterTarget
from conftest import bridge_config
from dashboard_stitch import stitch_all
from jobs_observability import jobs_overview_dataset
from node_observability import node_overview_dataset
from rl_observability import recent_rl_runs_dataset, rl_overview_dataset
from runs_observability import runs_overview_dataset
from server import create_app
from starlette.testclient import TestClient
from training_observability import training_overview_dataset
from zephyr_observability import zephyr_overview_dataset

ROOT = Path(__file__).resolve().parent.parent


def _source_and_queries(database: duckdb.DuckDBPyConnection):
    queries: list[str] = []
    query_limits: list[int] = []

    def query(sql: str, *, max_rows: int):
        queries.append(sql)
        query_limits.append(max_rows)
        table = database.execute(sql).fetch_arrow_table()
        if table.num_rows > max_rows:
            raise AssertionError(f"query returned {table.num_rows} rows with a {max_rows}-row cap")
        return table

    return (
        Record(
            target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
            query=query,
        ),
        queries,
        query_limits,
    )


def _app(source, *, max_rows: int = 1000):
    return create_app(replace(bridge_config(), max_rows=max_rows), {"marin": source}, {}, None, None, None)


def test_node_overview_serves_every_panel_from_one_source_query() -> None:
    database = duckdb.connect()
    database.execute(
        """CREATE TABLE "telemetry_v1.node_agent"(
               cluster VARCHAR, node_name VARCHAR, service VARCHAR, name VARCHAR,
               attributes_json VARCHAR, timestamp_ms BIGINT, seq BIGINT, value DOUBLE)"""
    )
    database.execute(
        """CREATE MACRO named_struct(k1, v1, k2, v2, k3, v3)
                   AS struct_pack(timestamp_ms := v1, seq := v2, value := v3)"""
    )
    gpu = json.dumps(
        {
            "gpu_index": "0",
            "gpu_uuid": "GPU-0",
            "pci_bus_id": "0000:01:00.0",
            "gpu_model": "H100",
            "driver_version": "1",
            "device_kind": "gpu",
            "source_replica_uid": "node-a",
        }
    )
    rows: list[tuple[int, str, float]] = []
    for timestamp in (0, 15_000, 30_000):
        rows.extend(
            (timestamp, name, value)
            for name, value in [
                ("gpu_power_watts", 400),
                ("gpu_utilization_percent", 80),
                ("gpu_sm_active_ratio", 0.5),
                ("gpu_tensor_active_ratio", 0.25),
                ("gpu_temperature_celsius", 70),
                ("gpu_memory_temperature_celsius", 75),
                ("gpu_memory_used_bytes", 40),
                ("gpu_memory_total_bytes", 80),
                ("node_cpu_utilization_percent", 60),
                ("node_memory_used_bytes", 30),
                ("node_memory_total_bytes", 60),
                ("node_disk_used_bytes", 20),
                ("node_disk_total_bytes", 80),
                ("node_network_receive_bytes", timestamp * 10),
                ("node_network_transmit_bytes", timestamp * 20),
                ("hardware_inventory", 1),
            ]
        )
    database.executemany(
        "INSERT INTO \"telemetry_v1.node_agent\" VALUES ('cw-a', 'node-a', 'iris-node-agent', ?, ?, ?, ?, ?)",
        [(name, gpu, timestamp, seq, value) for seq, (timestamp, name, value) in enumerate(rows)],
    )
    database.executemany(
        "INSERT INTO \"telemetry_v1.node_agent\" VALUES ('cw-a', 'node-a', 'iris-node-agent', ?, ?, 30000, ?, ?)",
        [
            ("gpu_nvlink_receive_bytes_per_second", gpu, 10_001, 10),
            ("gpu_nvlink_transmit_bytes_per_second", gpu, 10_002, 20),
            ("gpu_pcie_receive_bytes_per_second", gpu, 10_003, 30),
            ("gpu_pcie_transmit_bytes_per_second", gpu, 10_004, 40),
            ("gpu_xid_error_code", gpu, 10_005, 0),
            ("gpu_row_remap_failures", gpu, 10_006, 0),
            ("gpu_pcie_replay_errors", gpu, 10_007, 0),
        ],
    )
    source, queries, _ = _source_and_queries(database)
    params = {
        "clusters": "cw-a",
        "nodes": "node-a",
        "from": 0,
        "to": 45_000,
        "bucket_ms": 15_000,
    }
    sections = node_overview_dataset(("cw-a",), ("node-a",), 0, 45_000, 15_000).views

    with TestClient(_app(source)) as client:
        results = {
            view: client.get("/finelog/marin/v1/node/overview", params={**params, "view": view}) for view in sections
        }

    assert len(queries) == 1
    assert all(response.status_code == 200 for response in results.values())
    assert {row["value"] for row in results["power"].json()} == {400}
    assert {row["value"] for row in results["memory"].json()} == {50}
    assert results["inventory"].json() == [
        {
            "section": "inventory",
            "gpu": "0",
            "uuid": "GPU-0",
            "pci_bus": "0000:01:00.0",
            "model": "H100",
            "driver": "1",
            "lag": 15.0,
        }
    ]
    assert results["faults"].json() == []


def test_zephyr_overview_serves_every_panel_from_one_ranked_snapshot() -> None:
    database = duckdb.connect()
    database.execute(
        """CREATE TABLE "zephyr.shuffle"(
               execution_id VARCHAR, stage_name VARCHAR, target_shard BIGINT, num_targets BIGINT,
               input_rows BIGINT, payload_bytes BIGINT, num_sources BIGINT, attempt BIGINT,
               ts TIMESTAMP, seq BIGINT)"""
    )
    database.execute(
        "CREATE MACRO to_timestamp_millis(ms) AS TIMESTAMP '1970-01-01 00:00:00' + ms * INTERVAL 1 MILLISECOND"
    )
    database.execute(
        """INSERT INTO "zephyr.shuffle" VALUES
           ('execution', 'reduce', 0, 2, NULL, NULL, NULL, 0, TIMESTAMP '1970-01-01 00:00:01', 1),
           ('execution', 'reduce', 0, 2, 10, 100, 3, 1, TIMESTAMP '1970-01-01 00:00:02', 2),
           ('execution', 'reduce', 1, 2, NULL, NULL, NULL, 0, TIMESTAMP '1970-01-01 00:00:01', 3)"""
    )
    source, queries, query_limits = _source_and_queries(database)
    params = {"execution_id": "execution", "stage_name": "reduce", "from": 0, "to": 10_000}
    sections = zephyr_overview_dataset("execution", "reduce", 0, 10_000).views

    with TestClient(_app(source)) as client:
        results = {
            view: client.get("/finelog/marin/v1/zephyr/overview", params={**params, "view": view}) for view in sections
        }

    assert len(queries) == 1
    assert all(response.status_code == 200 for response in results.values())
    assert results["rows"].json() == [{"section": "rows", "target_reducer": "0", "input_rows": 10}]
    assert results["coverage"].json() == [
        {
            "section": "coverage",
            "observed_targets": 1,
            "expected_targets": 2,
            "unreported_targets": 1,
            "reported_empty_targets": 0,
            "observed_input_rows": 10,
        }
    ]
    assert [row["status"] for row in results["reducers"].json()] == ["REPORTED", "UNREPORTED"]

    with TestClient(_app(source, max_rows=3)) as client:
        capped = client.get("/finelog/marin/v1/zephyr/overview", params={**params, "view": "coverage"})

    assert capped.status_code == 400
    assert query_limits[-1] == 3


def test_priority_dashboards_use_only_bounded_panel_endpoints() -> None:
    dashboards = stitch_all(ROOT / "dashboards", ROOT / "dashboards" / "panels")
    start_ms, end_ms = 0, 60_000
    sections = {
        "node": frozenset(node_overview_dataset(("cw-a",), ("node-a",), start_ms, end_ms, 15_000).views),
        "zephyr": frozenset(zephyr_overview_dataset("execution", "stage", start_ms, end_ms).views),
        "training": frozenset(training_overview_dataset("run", start_ms, end_ms, 15_000).views),
        "runs": frozenset(runs_overview_dataset(("cw-a",), ("run",), start_ms, end_ms, 15_000).views),
        "rl": frozenset(rl_overview_dataset(("cw-a",), "run", start_ms, end_ms, 15_000).views),
        "accelerator": frozenset(accelerator_overview_dataset(("cw-a",), start_ms, end_ms, 15_000).views),
        "jobs": frozenset(jobs_overview_dataset(("cw-a",), ("job",), start_ms, end_ms, 15_000).views),
        "recent_rl": frozenset(recent_rl_runs_dataset(start_ms, end_ms).views),
    }
    expected = {
        "nodes.json": {"/v1/node/overview": (9, sections["node"])},
        "zephyr.json": {"/v1/zephyr/overview": (4, sections["zephyr"])},
        "training.json": {"/v1/training/overview": (16, sections["training"])},
        "runs.json": {"/v1/runs/overview": (8, sections["runs"])},
        "rl_runs.json": {"/v1/rl/overview": (15, sections["rl"])},
        "jobs.json": {"/v1/jobs/overview": (17, sections["jobs"])},
        "accelerators.json": {"/v1/accelerator/overview": (18, sections["accelerator"])},
        "home.json": {
            "/v1/accelerator/overview": (4, sections["accelerator"]),
            "/v1/jobs/overview": (4, sections["jobs"]),
            "/v1/rl/recent": (1, sections["recent_rl"]),
        },
    }

    for filename, endpoints in expected.items():
        targets = [
            target
            for panel in dashboards[filename]["panels"]
            for nested in (panel, *panel.get("panels", []))
            for target in nested.get("targets", [])
        ]
        assert all(target.get("url") != "/query" for target in targets), filename
        for endpoint, (count, sections) in endpoints.items():
            endpoint_targets = [target for target in targets if target.get("url") == endpoint]
            assert len(endpoint_targets) == count, filename
            dataset_keys = {
                tuple(
                    (param["key"], param["value"]) for param in target["url_options"]["params"] if param["key"] != "view"
                )
                for target in endpoint_targets
            }
            assert len(dataset_keys) == 1, filename
            views = {
                param["value"]
                for target in endpoint_targets
                for param in target["url_options"]["params"]
                if param["key"] == "view"
            }
            assert views <= sections, filename


def test_domain_source_counts_stay_within_the_declared_budget() -> None:
    start_ms, end_ms = 0, 60_000
    datasets = {
        "node": node_overview_dataset(("cw-a",), ("node-a",), start_ms, end_ms, 15_000),
        "zephyr": zephyr_overview_dataset("execution", "stage", start_ms, end_ms),
        "training": training_overview_dataset("run", start_ms, end_ms, 15_000),
        "runs": runs_overview_dataset(("cw-a",), ("run",), start_ms, end_ms, 15_000),
        "rl": rl_overview_dataset(("cw-a",), "run", start_ms, end_ms, 15_000),
        "accelerator": accelerator_overview_dataset(("cw-a",), start_ms, end_ms, 15_000),
        "jobs": jobs_overview_dataset(("cw-a",), ("job",), start_ms, end_ms, 15_000),
        "recent_rl": recent_rl_runs_dataset(start_ms, end_ms),
    }
    assert {name: len(dataset.sources) for name, dataset in datasets.items()} == {
        "node": 1,
        "zephyr": 1,
        "training": 3,
        "runs": 2,
        "rl": 3,
        "accelerator": 3,
        "jobs": 5,
        "recent_rl": 1,
    }
