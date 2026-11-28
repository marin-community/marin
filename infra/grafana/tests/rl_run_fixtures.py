# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import duckdb
import pytest
from conftest import (
    dashboard_panels,
    duckdb_bridge_app,
    finelog_sql_for_duckdb,
    install_finelog_dialect_macros,
    queried_namespace,
)
from dashboard_stitch import stitch_all
from rl_observability import rl_gpu_dataset, rl_overview_dataset
from rl_producers import collect_producers
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent

DASHBOARDS = ROOT / "dashboards"

NOW = datetime(2026, 8, 20, 12, tzinfo=UTC)

WINDOW_START = NOW - timedelta(hours=1)

CLUSTER = "cw-rno2a"

RUN_ID = "snowball-e6-muonh-0"

ASYNC_RUN_ID = "snowball-e6-muonh-0-async"

UNSTAMPED_RUN_ID = "snowball-e5-unstamped-0"

_NOW_MS = round(NOW.timestamp() * 1000)

_WINDOW_START_MS = round(WINDOW_START.timestamp() * 1000)

JOB_ID = "/atqamar/snowball-e6-muonh-0-attempt-0"

NODES = ("gb200-node-0", "gb200-node-1")

DEFAULT_EXECUTION_UID = "iris:/atqamar/snowball-e6-muonh-0-attempt-0/0:attempt:0"


class RunTelemetryRow(NamedTuple):
    cluster: str
    service: str
    run_id: str | None
    job_id: str | None
    execution_uid: str
    node_name: str | None
    process_index: str | None
    name: str
    value: float
    timestamp_ms: int
    seq: int
    resource_attributes_json: str
    attributes_json: str


_COLUMNS = RunTelemetryRow._fields


def _millis(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _row(
    *,
    service: str,
    name: str,
    value: float,
    moment: datetime,
    seq: int,
    run_id: str | None = None,
    job_id: str | None = None,
    node_name: str | None = None,
    execution_uid: str = DEFAULT_EXECUTION_UID,
    role: str = "",
    training_type: str | None = "sync",
    attributes: dict[str, str] | None = None,
) -> RunTelemetryRow:
    resource = {"role": role} if role else {}
    if training_type is not None:
        resource["training_type"] = training_type
    return RunTelemetryRow(
        CLUSTER,
        service,
        run_id,
        job_id,
        execution_uid,
        node_name,
        # Production stamps no process_index: it is NULL on every row of a real capture, so the
        # counter windows have to separate replicas without it.
        None,
        name,
        value,
        _millis(moment),
        seq,
        json.dumps(resource),
        json.dumps(attributes or {}),
    )


def _trainer_rows(bucket: int, moment: datetime) -> list[RunTelemetryRow]:
    rows = []
    for node in NODES:
        rows.append(
            _row(
                service="marinskyrl",
                name="policy_step",
                value=float(bucket),
                moment=moment,
                seq=bucket,
                run_id=RUN_ID,
                job_id=JOB_ID,
                node_name=node,
                role="trainer",
            )
        )
    for phase, seconds, outcome in (
        ("rollout_or_inference_wait", 44.0, "success"),
        ("train_step", 6.0, "success"),
        # A step that raised partway through. Its duration is a different quantity.
        ("train_step", 0.5, "failure"),
    ):
        rows.append(
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=seconds,
                moment=moment,
                seq=bucket,
                run_id=RUN_ID,
                job_id=JOB_ID,
                node_name=NODES[0],
                role="trainer",
                attributes={"phase": phase, "clock_domain": "critical_path", "outcome": outcome},
            )
        )
    for work_kind, count in (("rollout", 64.0), ("sample", 512.0), ("generated_token", 131072.0)):
        rows.append(
            _row(
                service="marinskyrl",
                name="work_completed",
                value=count,
                moment=moment,
                seq=bucket,
                run_id=RUN_ID,
                job_id=JOB_ID,
                node_name=NODES[0],
                role="trainer",
                attributes={"work_kind": work_kind},
            )
        )
    return rows


def _ray_rows(bucket: int, moment: datetime) -> list[RunTelemetryRow]:
    rows = []
    for node in NODES:
        for name, value in (
            ("ray_object_store_used_memory", 3.0e9),
            ("ray_object_store_available_memory", 1.0e9),
        ):
            rows.append(
                _row(
                    service="marinskyrl",
                    name=name,
                    value=value,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=node,
                    role="controller",
                    attributes={"metric_source": "ray", "source_temporality": "current_snapshot"},
                )
            )
        for state, value in (("Spilled", 2.0e9), ("Restored", 5.0e8)):
            rows.append(
                _row(
                    service="marinskyrl",
                    name="ray_spill_manager_objects_bytes",
                    value=value,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=node,
                    role="controller",
                    attributes={
                        "metric_source": "ray",
                        "source_temporality": "current_snapshot",
                        "state": state,
                    },
                )
            )
    # A cumulative counter from the same allowlist, which must never be averaged in.
    rows.append(
        _row(
            service="marinskyrl",
            name="ray_spill_manager_objects_bytes",
            value=9.9e12,
            moment=moment,
            seq=bucket,
            run_id=RUN_ID,
            job_id=JOB_ID,
            node_name=NODES[0],
            role="controller",
            attributes={
                "metric_source": "ray",
                "source_temporality": "cumulative_snapshot",
                "state": "Spilled",
            },
        )
    )
    return rows


def _node_rows(bucket: int, moment: datetime) -> list[RunTelemetryRow]:
    rows = []
    for node in NODES:
        for gpu in range(2):
            gauges = {
                "gpu_utilization_percent": 71.0 + bucket,
                "gpu_sm_active_ratio": 0.5 + gpu * 0.2,
                "gpu_tensor_active_ratio": 0.2 + gpu * 0.2,
                "gpu_memory_used_bytes": 40e9 + gpu * 20e9,
                "gpu_nvlink_receive_bytes_per_second": 100,
                "gpu_pcie_receive_bytes_per_second": 10,
                "gpu_power_watts": 620 + gpu * 10,
                "gpu_pcie_replay_errors": (10, 13, 1, 1, 2, 2)[bucket] if (node, gpu) == (NODES[0], 0) else 4,
            }
            for name, value in gauges.items():
                rows.append(
                    _row(
                        service="iris-node-agent",
                        name=name,
                        value=value,
                        moment=moment,
                        seq=bucket,
                        node_name=node,
                        attributes={"gpu_uuid": f"GPU-{node}-{gpu}"},
                    )
                )
    return rows


def _engine_rows(bucket: int, moment: datetime) -> list[RunTelemetryRow]:
    rows = []
    for engine in ("0", "1"):
        for name, value in (
            ("generation_tokens_total", 1024.0 * (bucket + 1)),
            ("prompt_tokens_total", 256.0 * (bucket + 1)),
            ("num_requests_running", 48.0),
            ("num_requests_waiting", 12.0),
            ("kv_cache_usage_perc", 0.71),
            ("prefix_cache_hits_total", 42.0 * (bucket + 1)),
            ("prefix_cache_queries_total", 100.0 * (bucket + 1)),
            ("num_preemptions_total", 2.0 * (bucket + 1)),
        ):
            rows.append(
                _row(
                    service="vllm",
                    name=name,
                    value=value,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[1],
                    role="inference",
                    attributes={"metric_source": "vllm", "engine": engine},
                )
            )
        for stage in (
            "request_queue_time_seconds_sum",
            "request_queue_time_seconds_count",
            "request_decode_time_seconds_sum",
            "request_decode_time_seconds_count",
        ):
            rows.append(
                _row(
                    service="vllm",
                    name=stage,
                    value=(3.5 if stage.endswith("_sum") else 1.0) * (bucket + 1),
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[1],
                    role="inference",
                    attributes={"metric_source": "vllm", "engine": engine},
                )
            )
        for reason, value in (
            ("length", 8.0 * (bucket + 1)),
            ("stop", 40.0 * (bucket + 1)),
            ("abort", 1.0 * (bucket + 1)),
        ):
            rows.append(
                _row(
                    service="vllm",
                    name="request_success_total",
                    value=value,
                    moment=moment,
                    seq=bucket,
                    run_id=RUN_ID,
                    job_id=JOB_ID,
                    node_name=NODES[1],
                    role="inference",
                    attributes={"metric_source": "vllm", "engine": engine, "finished_reason": reason},
                )
            )
    return rows


def _run_rows() -> list[RunTelemetryRow]:
    rows = []
    for bucket in range(6):
        moment = WINDOW_START + timedelta(minutes=5 * bucket)
        rows.extend(_trainer_rows(bucket, moment))
        rows.extend(_ray_rows(bucket, moment))
        rows.extend(_node_rows(bucket, moment))
        rows.extend(_engine_rows(bucket, moment))
    return rows


_SEMANTIC_STREAM = {
    "vllm": "telemetry_v1.vllm",
    "iris-node-agent": "telemetry_v1.node_agent",
    "marinskyrl": "telemetry_v1.marinskyrl",
    "harbor": "telemetry_v1.harbor",
}

_SCHEMA = """(
    cluster VARCHAR,
    service VARCHAR,
    run_id VARCHAR,
    job_id VARCHAR,
    execution_uid VARCHAR,
    node_name VARCHAR,
    process_index VARCHAR,
    name VARCHAR,
    value DOUBLE,
    timestamp_ms BIGINT,
    seq BIGINT,
    resource_attributes_json VARCHAR,
    attributes_json VARCHAR
)"""


def _fresh_store() -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    for stream in sorted(set(_SEMANTIC_STREAM.values())):
        database.execute(f'CREATE TABLE "{stream}"{_SCHEMA}')
    install_finelog_dialect_macros(database)
    return database


def _seed_rows(database, rows: list[tuple]) -> None:
    placeholders = ", ".join("?" for _ in _COLUMNS)
    service_index = _COLUMNS.index("service")
    routed: dict[str, list] = {}
    for row in rows:
        routed.setdefault(_SEMANTIC_STREAM[row[service_index]], []).append(row)
    for stream, stream_rows in routed.items():
        database.executemany(f'INSERT INTO "{stream}" VALUES ({placeholders})', stream_rows)


@pytest.fixture
def rl_run_store() -> duckdb.DuckDBPyConnection:
    database = _fresh_store()
    _seed_rows(database, _run_rows())
    return database


def _view_sql(view: str, endpoint: str) -> str:
    build = {"/v1/rl/overview": rl_overview_dataset, "/v1/rl/gpu": rl_gpu_dataset}[endpoint]
    dataset = build((CLUSTER,), RUN_ID, _WINDOW_START_MS, _NOW_MS, 5 * 60 * 1000)
    sources = ",\n".join(f"{source.name} AS ({source.sql})" for source in dataset.sources)
    return finelog_sql_for_duckdb(f"WITH {sources}\n{dataset.views[view]}")


def _panel_sql(title: str) -> str:
    (panel,) = [panel for panel in dashboard_panels(_dashboard()["panels"]) if panel.get("title") == title]
    (target,) = [target for target in panel["targets"] if target.get("url") in ("/v1/rl/overview", "/v1/rl/gpu")]
    (view,) = [param["value"] for param in target["url_options"]["params"] if param["key"] == "view"]
    return _view_sql(view, target["url"])


def _request_rows(rl_run_store, path, params):
    app = duckdb_bridge_app(rl_run_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{path}", params=params)
        assert response.status_code == 200, response.text
        return response.json()


def _overview_rows(rl_run_store, view, endpoint="/v1/rl/overview"):
    return _request_rows(
        rl_run_store,
        endpoint,
        {
            "clusters": CLUSTER,
            "run": RUN_ID,
            "from": _WINDOW_START_MS,
            "to": _NOW_MS,
            "bucket_ms": 300000,
            "view": view,
        },
    )


def _census(database, present: frozenset[str]) -> list[dict[str, object]]:
    """Return producer census rows from the deployment's available namespaces."""

    def query(sql: str) -> list[dict[str, object]]:
        assert queried_namespace(sql) in present, f"queried {queried_namespace(sql)}, which is absent"
        columns = [description[0] for description in database.execute(sql).description]
        return [dict(zip(columns, row, strict=True)) for row in database.execute(sql).fetchall()]

    return collect_producers(query, present, RUN_ID, (CLUSTER,), _WINDOW_START_MS, _NOW_MS)


def _dashboard() -> dict:
    return stitch_all(DASHBOARDS, DASHBOARDS / "panels")["rl_runs.json"]


def _recent_runs_panel() -> dict:
    home = stitch_all(DASHBOARDS, DASHBOARDS / "panels")["home.json"]
    (panel,) = [p for p in home["panels"] if p.get("title") == "Recent RL runs"]
    return panel
