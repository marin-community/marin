# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as Record

import duckdb
import pytest
from accelerator_observability import accelerator_overview_dataset
from async_rl_observability import async_rl_overview_dataset
from config import ClusterTarget
from conftest import bridge_config
from dashboard_dataset import DashboardDataset
from dashboard_stitch import stitch_all
from jobs_observability import jobs_overview_dataset
from node_observability import node_overview_dataset
from rl_observability import recent_rl_runs_dataset, rl_overview_dataset, rl_sync_generation_dataset
from runs_observability import runs_overview_dataset
from server import create_app
from starlette.testclient import TestClient
from training_observability import training_overview_dataset
from vllm_observability import VLLM_OVERVIEW_SECTIONS
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


def _materialize_sources(database: duckdb.DuckDBPyConnection, dataset: DashboardDataset, *names: str) -> None:
    selected = set(names)
    for source in dataset.sources:
        if source.name in selected:
            database.execute(f'CREATE TEMP TABLE "{source.name}" AS {source.sql}')


def _result_rows(database: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, object]]:
    result = database.execute(sql)
    columns = [description[0] for description in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


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
        "rl_sync_generation": frozenset(rl_sync_generation_dataset(("cw-a",), "run", start_ms, end_ms, 15_000).views),
        "vllm": VLLM_OVERVIEW_SECTIONS,
        "accelerator": frozenset(accelerator_overview_dataset(("cw-a",), start_ms, end_ms, 15_000).views),
        "jobs": frozenset(jobs_overview_dataset(("cw-a",), ("job",), start_ms, end_ms, 15_000).views),
        "recent_rl": frozenset(recent_rl_runs_dataset(start_ms, end_ms).views),
        "async_rl": frozenset(
            async_rl_overview_dataset(("cw-a",), "run", "job", ("execution",), start_ms, end_ms, 15_000).views
        ),
    }
    expected = {
        "nodes.json": {"/v1/node/overview": (9, sections["node"])},
        "zephyr.json": {"/v1/zephyr/overview": (4, sections["zephyr"])},
        "training.json": {"/v1/training/overview": (16, sections["training"])},
        "runs.json": {"/v1/runs/overview": (8, sections["runs"])},
        "rl_runs.json": {"/v1/rl/overview": (19, sections["rl"])},
        "rl_sync_generation.json": {
            "/v1/rl/generation": (5, sections["rl_sync_generation"]),
            "/v1/vllm/overview": (6, sections["vllm"]),
        },
        "async_rl.json": {"/v1/async-rl/overview": (48, sections["async_rl"])},
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
        "rl_sync_generation": rl_sync_generation_dataset(("cw-a",), "run", start_ms, end_ms, 15_000),
        "accelerator": accelerator_overview_dataset(("cw-a",), start_ms, end_ms, 15_000),
        "jobs": jobs_overview_dataset(("cw-a",), ("job",), start_ms, end_ms, 15_000),
        "recent_rl": recent_rl_runs_dataset(start_ms, end_ms),
        "async_rl": async_rl_overview_dataset(("cw-a",), "run", "job", ("execution",), start_ms, end_ms, 15_000),
    }
    assert {name: len(dataset.sources) for name, dataset in datasets.items()} == {
        "node": 1,
        "zephyr": 1,
        "training": 3,
        "runs": 2,
        "rl": 4,
        "rl_sync_generation": 1,
        "accelerator": 3,
        "jobs": 5,
        "recent_rl": 1,
        "async_rl": 9,
    }


def test_accelerator_sources_preserve_stale_devices_and_aggregate_each_gpu_once() -> None:
    database = duckdb.connect()
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
    rows = []
    for gpu, watts in (("GPU-0", 100.0), ("GPU-1", 200.0)):
        attributes = json.dumps({"gpu_uuid": gpu, "gpu_index": gpu[-1], "gpu_model": "H100"})
        rows.extend(
            [
                ("cw-a", "node-a", "iris-node-agent", "gpu_power_watts", attributes, 0, watts),
                ("cw-a", "node-a", "iris-node-agent", "gpu_power_watts", attributes, 1_000, watts),
                ("cw-a", "node-a", "iris-node-agent", "hardware_inventory", attributes, 900_000, 1.0),
            ]
        )
    database.executemany('INSERT INTO "telemetry_v1.node_agent" VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
    database.execute("INSERT INTO \"levanter.metrics\" VALUES ('cw-a', 'node-a', 'run-a', 1, 0)")
    dataset = accelerator_overview_dataset(("cw-a",), 0, 1_200_000, 30_000)
    _materialize_sources(database, dataset, "devices", "attribution")

    freshness = _result_rows(database, dataset.views["freshness"])
    model_power = _result_rows(database, dataset.views["model_power"])
    run_power = _result_rows(database, dataset.views["run_power"])

    assert freshness == [{"cluster": "cw-a", "gpus": 2, "nodes": 1, "lag_seconds": 1199.0}]
    assert model_power[0]["value"] == pytest.approx(0.3)
    assert run_power[0]["value"] == pytest.approx(0.3)


def test_jobs_views_keep_selected_jobs_and_independent_top_twenty_rankings() -> None:
    database = duckdb.connect()
    database.execute(
        """CREATE TABLE "iris.task_state"(
               ts TIMESTAMP, cluster VARCHAR, root_job_id VARCHAR,
               pending BIGINT, assigned BIGINT, building BIGINT, running BIGINT,
               oldest_pending_age_ms BIGINT, oldest_building_age_ms BIGINT)"""
    )
    database.execute(
        """CREATE TABLE "iris.task"(
               ts TIMESTAMP, cluster VARCHAR, task_id VARCHAR,
               memory_mb DOUBLE, cpu_millicores DOUBLE)"""
    )
    database.execute(
        """CREATE TABLE "iris.worker"(
               ts TIMESTAMP, cluster VARCHAR, worker_id VARCHAR, cpu_pct DOUBLE, mem_bytes DOUBLE)"""
    )
    database.execute(
        "CREATE MACRO to_timestamp_millis(ms) AS TIMESTAMP '1970-01-01 00:00:00' + ms * INTERVAL 1 MILLISECOND"
    )
    database.execute("CREATE MACRO date_bin(width, moment) AS time_bucket(width, moment)")
    database.executemany(
        'INSERT INTO "iris.task_state" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
            ("1970-01-01 00:00:01", "cw-a", "selected", 1, 0, 0, 1, 0, 0),
            ("1970-01-01 00:00:01", "cw-a", "unrelated-stuck", 1, 0, 0, 0, 1_000_000, 0),
        ],
    )
    database.executemany(
        'INSERT INTO "iris.task" VALUES (?, ?, ?, ?, ?)',
        [("1970-01-01 00:00:01", "cw-a", f"task-{index:02}", 21 - index, index) for index in range(21)]
        + [("1970-01-01 00:00:01", "cw-b", "task-other-cluster", 1_000, 1_000)],
    )
    database.executemany(
        'INSERT INTO "iris.worker" VALUES (?, ?, ?, ?, ?)',
        [("1970-01-01 00:00:01", "cw-a", f"worker-{index:02}", index, 21 - index) for index in range(21)]
        + [("1970-01-01 00:00:01", "cw-b", "worker-other-cluster", 1_000, 1_000)],
    )
    dataset = jobs_overview_dataset(("cw-a",), ("selected",), 0, 60_000, 15_000)
    _materialize_sources(database, dataset, "task_state", "tasks", "workers")

    assert [row["job"] for row in _result_rows(database, dataset.views["active_jobs"])] == ["selected"]
    assert _result_rows(database, dataset.views["stuck_jobs"]) == [{"stuck": 1}]
    assert {row["series"] for row in _result_rows(database, dataset.views["task_memory"])} == {
        f"task-{index:02}" for index in range(20)
    }
    assert {row["series"] for row in _result_rows(database, dataset.views["task_cpu"])} == {
        f"task-{index:02}" for index in range(1, 21)
    }
    assert {row["series"] for row in _result_rows(database, dataset.views["worker_cpu"])} == {
        f"worker-{index:02}" for index in range(1, 21)
    }
    assert {row["series"] for row in _result_rows(database, dataset.views["worker_memory"])} == {
        f"worker-{index:02}" for index in range(20)
    }


def test_runs_views_use_run_wide_cardinality_training_freshness_and_global_node_ownership() -> None:
    database = duckdb.connect()
    database.execute(
        """CREATE TABLE "levanter.metrics"(
               cluster VARCHAR, node_name VARCHAR, run_id VARCHAR, step BIGINT,
               process_index VARCHAR, name VARCHAR, value DOUBLE, timestamp_ms BIGINT)"""
    )
    database.execute(
        """CREATE TABLE "telemetry_v1.node_agent"(
               cluster VARCHAR, node_name VARCHAR, service VARCHAR, name VARCHAR,
               attributes_json VARCHAR, value DOUBLE, timestamp_ms BIGINT)"""
    )
    database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
    database.executemany(
        'INSERT INTO "levanter.metrics" VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        [
            ("cw-a", "node-a", "selected", 1, "0", "train_loss", 3.0, 0),
            ("cw-a", "node-b", "selected", 2, "1", "train_loss", 2.0, 20_000),
            ("cw-a", "node-c", "selected", 2, "2", "eval_loss", 1.0, 90_000),
            ("cw-a", "node-a", "other", 1, "0", "train_loss", 4.0, 1_000),
            ("cw-a", "node-a", "other", 1, "1", "throughput_mfu", 0.5, 2_000),
            ("cw-a", "node-a", "other", 1, "2", "throughput_tokens_per_second", 10.0, 3_000),
        ],
    )
    database.executemany(
        'INSERT INTO "telemetry_v1.node_agent" VALUES (?, ?, ?, ?, ?, ?, ?)',
        [
            ("cw-a", "node-a", "iris-node-agent", "gpu_power_watts", json.dumps({"gpu_uuid": "GPU-0"}), 100, 0),
            ("cw-a", "node-a", "iris-node-agent", "gpu_power_watts", json.dumps({"gpu_uuid": "GPU-1"}), 200, 0),
        ],
    )
    dataset = runs_overview_dataset(("cw-a",), ("selected",), 0, 120_000, 15_000)
    _materialize_sources(database, dataset, "metrics", "power")

    active = _result_rows(database, dataset.views["active"])

    assert active[0]["nodes"] == 2
    assert active[0]["processes"] == 2
    assert active[0]["sample_age_seconds"] == 100.0
    assert _result_rows(database, dataset.views["power"]) == []
