# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import duckdb
import pyarrow as pa
import pytest
from conftest import dashboard_panels, finelog_sql_for_duckdb, install_finelog_dialect_macros
from dashboard_stitch import stitch_all
from rl_execution_observability import (
    rl_execution_dataset,
)

GRAFANA_DIR = Path(__file__).parents[1]

DASHBOARD = stitch_all(GRAFANA_DIR / "dashboards", GRAFANA_DIR / "dashboards/panels")["rl_runs.json"]

ENDPOINT = "/v1/rl/execution"

PANELS = {
    panel["title"]: panel
    for panel in dashboard_panels(DASHBOARD["panels"])
    if any(target["url"] == ENDPOINT for target in panel.get("targets", []))
}

TABLE = '"telemetry_v1.marinskyrl"'

SERVICE = "marinskyrl"

CLUSTER = "cw-us-east-02a"

RUN_ID = "run"

JOB_ID = "job"

DRIVER = "driver"

WORKER = "worker"

WINDOW_START_MS = 1788566400000

WINDOW_MS = 300000

BASE_EPOCH_MS = WINDOW_START_MS + 60000


def _window_literal(epoch_ms):
    return f"TIMESTAMP '{datetime.fromtimestamp(epoch_ms / 1000, UTC).strftime('%Y-%m-%d %H:%M:%S')}'"


def resolve(sql):
    for macro, value in {
        "{{from}}": _window_literal(WINDOW_START_MS),
        "{{to}}": _window_literal(WINDOW_START_MS + WINDOW_MS),
        "${training_type:sqlstring}": "'async'",
        "${cluster:sqlstring}": f"'{CLUSTER}'",
        "${run:sqlstring}": f"'{RUN_ID}'",
        "${job:sqlstring}": f"'{JOB_ID}'",
        "${execution:sqlstring}": f"'{DRIVER}','{WORKER}'",
    }.items():
        sql = sql.replace(macro, value)
    return sql


def run_variable_sql():
    variable = next(item for item in DASHBOARD["templating"]["list"] if item["name"] == "run")
    params = variable["query"]["infinityQuery"]["url_options"]["params"]
    return next(param["value"] for param in params if param["key"] == "sql")


def dataset(*, window_ms=WINDOW_MS, bucket_ms=WINDOW_MS):
    """The dataset every panel requests for the selected identity over this suite's window."""
    return rl_execution_dataset(
        (CLUSTER,), RUN_ID, JOB_ID, (DRIVER, WORKER), WINDOW_START_MS, WINDOW_START_MS + window_ms, bucket_ms
    )


def query(database, title, *, window_ms=WINDOW_MS, bucket_ms=WINDOW_MS):
    target = PANELS[title]["targets"][0]
    selected = dataset(window_ms=window_ms, bucket_ms=bucket_ms)
    for source in selected.sources:
        database.execute(f'CREATE OR REPLACE TEMP TABLE "{source.name}" AS {finelog_sql_for_duckdb(source.sql)}')
    view = next(param["value"] for param in target["url_options"]["params"] if param["key"] == "view")
    cursor = database.execute(selected.views[view])
    columns = [column[0] for column in cursor.description]
    assert columns == [column["selector"] for column in target["columns"]]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class TelemetryRow(NamedTuple):
    """One row in the finelog schema, in the column order the table declares."""

    cluster: str
    service: str
    run_id: str | None
    job_id: str | None
    execution_uid: str | None
    timestamp_ms: int
    seq: int
    name: str
    value: float
    attributes_json: str
    resource_attributes_json: str
    body_json: str


SCHEMA = pa.schema(
    [
        ("cluster", pa.string()),
        ("service", pa.string()),
        ("run_id", pa.string()),
        ("job_id", pa.string()),
        ("execution_uid", pa.string()),
        ("timestamp_ms", pa.int64()),
        ("seq", pa.int64()),
        ("name", pa.string()),
        ("value", pa.float64()),
        ("attributes_json", pa.string()),
        ("resource_attributes_json", pa.string()),
        ("body_json", pa.string()),
    ]
)


def seed(database, rows):
    # DuckDB's executemany costs milliseconds a row; one Arrow batch costs microseconds.
    database.register("seeded_rows", pa.table([list(column) for column in zip(*rows, strict=True)], schema=SCHEMA))
    database.execute(f"INSERT INTO {TABLE} SELECT * FROM seeded_rows")


def telemetry_row(
    name,
    value=0,
    *,
    job=JOB_ID,
    execution=DRIVER,
    timestamp=BASE_EPOCH_MS,
    seq=0,
    attributes=None,
    resource=None,
    body=None,
) -> TelemetryRow:
    """One row the selected run, job and execution reported."""
    return TelemetryRow(
        cluster=CLUSTER,
        service=SERVICE,
        run_id=RUN_ID,
        job_id=job,
        execution_uid=execution,
        timestamp_ms=timestamp,
        seq=seq,
        name=name,
        value=value,
        attributes_json=json.dumps(attributes or {}),
        resource_attributes_json=json.dumps(resource or {}),
        body_json=json.dumps(body or {}),
    )


@pytest.fixture
def telemetry_table():
    with duckdb.connect() as database:
        database.register("finelog_schema", SCHEMA.empty_table())
        database.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM finelog_schema")
        install_finelog_dialect_macros(database)
        yield database


@pytest.fixture
def rl_execution_store(telemetry_table):
    database = telemetry_table
    rows = []

    def add(name, value=0, *, attributes=None, body=None, process="trainer", execution=DRIVER, timestamp=None):
        rows.append(
            telemetry_row(
                name,
                value,
                execution=execution,
                timestamp=BASE_EPOCH_MS + len(rows) if timestamp is None else timestamp,
                seq=len(rows),
                attributes={"role": "trainer", "step": "1", **(attributes or {})},
                resource={"role": "trainer", "host": process, "training_type": "async"},
                body=body,
            )
        )

    add("lifecycle", body={"state": "started"})
    add("terminal", body={"status": "completed", "reason": "normal_exit"})
    add("policy_step", 1)
    add("weight_sync_completed", body={"model_version_step": 1})
    for kind, value in [("generated_token", 150), ("consumed_response_token", 100), ("consumed_loss_token", 90)]:
        add(
            "work_completed",
            value,
            attributes={"work_kind": kind, **({"weights_step": "0"} if kind == "generated_token" else {})},
        )
    for phase in (
        "step",
        "wait_for_generation_buffer",
        "run_training",
        "fwd_logprobs_values_reward",
        "train_critic_and_policy",
        "sync_weights",
        "init_weight_sync_state",
        "offload_policy_model_to_cpu",
    ):
        add(
            "phase_duration_seconds",
            2,
            attributes={
                "phase": phase,
                **({"clock_domain": "inclusive_wall", "root": "step", "parent": ""} if phase == "step" else {}),
            },
        )
    for value in (10, 30):
        add("rollout_wait_seconds", value, attributes={"wait": "slot", "stat": "sum"})
    for value in (2, 10):
        add("rollout_waits", value, attributes={"wait": "slot"})
    add("rollout_wait_seconds", 18, attributes={"wait": "slot", "stat": "max"})
    for value in (1, 5, 3):
        add("rollout_queue_depth", value)
    add("rollout_capacity", 64)
    for value in (0.1, 0.5, 9.0):
        add("rollout_buffer_dwell_seconds", value, attributes={"disposition": "consumed"})
    for value in (0, 1, 9):
        add("rollout_staleness_steps", value)
    for value in (0.5, 2.0, 30.0):
        add("phase_duration_seconds", value, attributes={"phase": "rollout_call", "outcome": "success"})
    add("phase_duration_seconds", -0.25, attributes={"phase": "rollout_call_residual", "outcome": "success"})
    add(
        "phase_duration_seconds",
        8,
        attributes={"phase": "ppo_train", "outcome": "success", "rank": "0", "backend": "megatron"},
        execution=WORKER,
    )
    add(
        "phase_duration_seconds",
        -0.5,
        attributes={"phase": "ppo_train_residual", "outcome": "success", "rank": "0", "backend": "megatron"},
        execution=WORKER,
    )
    add("event_loop_lag_seconds", 0.1)
    # The training window below spans BASE_EPOCH_MS to BASE_EPOCH_MS + 10 s; calls a and b
    # finish inside it, "outside" finishes after it.
    for call, finish, tokens in [("a", 5000, 7), ("b", 9000, 11), ("outside", 15000, 17)]:
        add(
            "rollout_call",
            attributes={"outcome": "success"},
            body={
                "call_id": call,
                "started_unix_ms": BASE_EPOCH_MS - 1000,
                "finished_unix_ms": BASE_EPOCH_MS + finish,
                "duration_seconds": (finish + 1000) / 1000,
                "response_tokens": tokens,
            },
        )
    for disposition, tokens in [("consumed", 100), ("epoch_discarded", 50)]:
        add("rollout_groups", 1, attributes={"disposition": disposition})
        add("rollout_group_tokens", tokens, attributes={"disposition": disposition})
    for metric, value in [
        ("reward/avg_raw_reward", 0.5),
        ("eval/all/avg_score", 0.25),
        ("eval/all/response_tokens_mean", 150),
        ("eval/all/response_tokens_max", 300),
        ("eval/all/length_stop_fraction", 0.5),
        ("eval/all/completed_stop_fraction", 0.5),
        ("eval/all/stop_reason_coverage", 1),
        ("eval/all/length_stop_score_contribution", 0.2),
        ("eval/all/completed_stop_score_contribution", 0.05),
        ("policy/policy_loss", -0.2),
        ("policy/raw_grad_norm", 4),
        ("consumed/length_stop_fraction", 0.25),
        ("consumed/stop_reason_coverage", 1),
        ("policy/mismatch/pooled/log_ratio_mean", -0.1),
        ("policy/mismatch/pooled/log_ratio_mean_squared", 0.04),
        ("policy/mismatch/pooled/log_ratio_abs_p99", 0.7),
        ("policy/mismatch/pooled/lower_clip_pressure", 0.2),
        ("policy/mismatch/pooled/upper_clip_pressure", 0.1),
        ("policy/mismatch/pooled/finite_fraction", 0.75),
        ("policy/mismatch/pooled/ess_fraction", 0.8),
        ("async/performance/core_seconds", 10),
        ("async/performance/cycle_seconds", 25),
        ("async/performance/consumed_loss_tokens_per_core_second", 100),
        ("async/performance/consumed_loss_tokens_per_cycle_second", 40),
        ("async/performance/buffer_wait_fraction", 0.2),
        ("async/performance/loss_tokens_per_configured_policy_gpu_second", 5),
        ("async/performance/configured_policy_gpus", 8),
        ("async/performance/configured_inference_gpus", 8),
    ]:
        add(
            "training_metric_value",
            value,
            attributes={"metric": metric, "payload_kind": "eval" if metric.startswith("eval/") else "train"},
        )
    add(
        "cuda_memory_observation",
        execution=WORKER,
        process="learner",
        attributes={
            "worker_role": "policy",
            "rank": "0",
            "gpu_uuid": "GPU-A",
            "phase": "ppo_train",
        },
        body={
            "peak_allocated_bytes": 4 * 2**30,
            "peak_reserved_bytes": 6 * 2**30,
            "allocated_bytes": 3 * 2**30,
            "device_free_bytes": 2**30,
            "device_total_bytes": 8 * 2**30,
        },
    )
    phase_start = BASE_EPOCH_MS
    for phase, start, finish in [("training", 0, 10000), ("weight_sync", 10000, 14000)]:
        add(
            "async_phase_window",
            attributes={"phase": phase, "outcome": "success"},
            body={
                "started_unix_ms": phase_start + start,
                "finished_unix_ms": phase_start + finish,
                "duration_seconds": (finish - start) / 1000,
            },
        )
    # Valid 0->4s (40 tokens); counter reset 4->6s excluded; valid 6->8s (20).
    # 8->12s crosses the weight-sync boundary: cannot attribute it to either phase.
    for engine, samples in [
        ("engine-A", [(0, 100), (4000, 140), (6000, 5), (8000, 25), (12000, 100)]),
        ("engine-B", [(0, 1000), (4000, 1080)]),
    ]:
        for offset, value in samples:
            add(
                "generation_tokens_total",
                value,
                timestamp=phase_start + offset,
                attributes={
                    "engine": engine,
                    "engine_index": "0",
                    "metric_source": "vllm",
                    "source_temporality": "cumulative_snapshot",
                    "step": str(offset),
                },
            )
    # Collector identity is part of the clock domain, even with the same engine label.
    for offset, value in [(0, 0), (4000, 99999)]:
        add(
            "generation_tokens_total",
            value,
            timestamp=phase_start + offset,
            process="other",
            attributes={"engine": "engine-A", "metric_source": "vllm", "source_temporality": "cumulative_snapshot"},
        )
    # Two optimizer steps, both within one display bucket: do not pool their staleness.
    for step, staleness, tokens in [(2, 0, 10), (2, 1, 30), (2, 1, 50), (3, 0, 20), (3, 1, 70)]:
        add("rollout_staleness_steps", staleness, attributes={"step": str(step)})
        # consumed_staleness carries body.staleness and body.response_tokens; older emitters lack it.
        add(
            "consumed_staleness",
            attributes={"step": str(step)},
            body={"staleness": staleness, "response_tokens": tokens},
        )
    for step, scale in [(2, 1), (3, 2)]:
        for metric, value in [
            ("policy/mismatch/staleness0/log_ratio_abs_mean", 0.1),
            ("policy/mismatch/staleness1/log_ratio_abs_mean", 0.3),
            ("policy/mismatch/staleness0/log_ratio_abs_p999", 0.7),
            ("policy/mismatch/staleness0/ess_fraction", 0.8),
            ("policy/log_ratio_abs_mean", 0.05),
            ("policy/log_ratio_ess_fraction", 0.9),
            ("policy/mismatch/pooled/pos_first256/log_ratio_abs_mean", 0.12),
            ("policy/mismatch/pooled/pos_last256/log_ratio_abs_mean", 0.23),
            ("policy/log_ratio_pos_first256/log_ratio_abs_mean", 0.01),
            ("policy/log_ratio_pos_last256/log_ratio_abs_mean", 0.04),
            ("policy/correction/weight_mean", 0.75),
            ("policy/correction/masked_fraction", 0.05),
            ("policy/correction/truncated_fraction", 0.12),
            ("policy/ppo_clip_ratio", 0.08),
        ]:
            add(
                "training_metric_value",
                value * scale,
                attributes={"metric": metric, "payload_kind": "train", "step": str(step)},
            )
    add("telemetry_lost_records", 0)
    add("telemetry_rejected_records", 0)
    for n, value in ((1, 0.75), (16, 0.25)):
        add("training_metric_value", value, attributes={"metric": f"reward/avg_pass_at_{n}", "payload_kind": "train"})
    # One copy of every row per predicate the panels filter on, each copy falsifying one
    # predicate and carrying a value no assertion below expects. The other run is a sync
    # run, which the run picker has to leave out of its dropdown as well.
    sync_resource = json.dumps({"role": "trainer", "host": "trainer", "training_type": "sync"})
    add("work_completed", 64, attributes={"work_kind": "rollout"})
    add("work_completed", 512, attributes={"work_kind": "sample"})
    distractors = []
    for row in rows:
        for column, replacement in (
            ("cluster", "cw-us-west-04b"),
            ("service", "levanter"),
            ("run_id", "other-run"),
            ("job_id", "other-job"),
            ("execution_uid", "other-execution"),
            ("execution_uid", None),
            ("timestamp_ms", WINDOW_START_MS - 1000),
        ):
            other = row._replace(**{column: replacement}, value=1000)
            if column == "run_id":
                other = other._replace(resource_attributes_json=sync_resource)
            distractors.append(other)
    sync_rows = []
    for row in rows:
        metric = json.loads(row.attributes_json).get("metric", "")
        if metric.startswith("policy/mismatch/staleness") and not metric.startswith("policy/mismatch/staleness0/"):
            continue
        resource = {**json.loads(row.resource_attributes_json), "training_type": "sync"}
        body = json.loads(row.body_json)
        if row.name == "consumed_staleness":
            body["staleness"] = 0
        sync_rows.append(
            row._replace(
                run_id="sync-run",
                resource_attributes_json=json.dumps(resource),
                body_json=json.dumps(body),
                value=0 if row.name == "rollout_staleness_steps" else row.value,
            )
        )
    seed(database, rows + distractors + sync_rows)
    return database
