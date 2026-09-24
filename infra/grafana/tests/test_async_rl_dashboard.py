# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as Record
from typing import NamedTuple

import duckdb
import pyarrow as pa
import pytest
from async_rl_observability import async_rl_overview_dataset
from config import ClusterTarget
from conftest import bridge_config, install_finelog_dialect_macros
from server import create_app
from starlette.testclient import TestClient

DASHBOARD = json.loads((Path(__file__).parents[1] / "dashboards/async_rl.json").read_text())
PANELS = {panel["title"]: panel for panel in DASHBOARD["panels"] if "targets" in panel}
ENDPOINT = "/v1/async-rl/overview"

# The finelog namespace every async panel reads, quoted the way the dashboards spell it.
TABLE = '"telemetry_v1.marinskyrl"'

# The identity the panels are asked about: one cluster, run, job and its two executions. Rows
# carrying any other value are distractors the predicates have to drop.
SERVICE = "marinskyrl"
CLUSTER = "cw-us-east-02a"
RUN_ID = "run"
JOB_ID = "job"
DRIVER = "driver"
WORKER = "worker"

# The selected window, and the instant seeded rows land on. Panels bound their scans by
# {{from}}/{{to}}, so rows must fall inside it for any panel to return them; tests that
# exercise clipping place their rows relative to these bounds rather than on a literal.
WINDOW_START_MS = 1788566400000
WINDOW_MS = 300000
BASE_EPOCH_MS = WINDOW_START_MS + 60000


def _window_literal(epoch_ms):
    return f"TIMESTAMP '{datetime.fromtimestamp(epoch_ms / 1000, UTC).strftime('%Y-%m-%d %H:%M:%S')}'"


def resolve(sql, *, window_ms=WINDOW_MS):
    for macro, value in {
        "{{from}}": _window_literal(WINDOW_START_MS),
        "{{to}}": _window_literal(WINDOW_START_MS + window_ms),
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


def panel_view(title):
    target = PANELS[title]["targets"][0]
    assert target["url"] == ENDPOINT
    return next(param["value"] for param in target["url_options"]["params"] if param["key"] == "view")


def dataset(*, window_ms=WINDOW_MS, bucket_ms=WINDOW_MS):
    """The dataset every panel requests for the selected identity over this suite's window."""
    return async_rl_overview_dataset(
        (CLUSTER,), RUN_ID, JOB_ID, (DRIVER, WORKER), WINDOW_START_MS, WINDOW_START_MS + window_ms, bucket_ms
    )


def materialize_sources(database, selected):
    for source in selected.sources:
        database.execute(f'CREATE OR REPLACE TEMP TABLE "{source.name}" AS {source.sql}')


def query(database, title, *, window_ms=WINDOW_MS, bucket_ms=WINDOW_MS):
    target = PANELS[title]["targets"][0]
    selected = dataset(window_ms=window_ms, bucket_ms=bucket_ms)
    materialize_sources(database, selected)
    cursor = database.execute(selected.views[panel_view(title)])
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
def store(telemetry_table):
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
        add("work_completed", value, attributes={"work_kind": kind})
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
        add("phase_duration_seconds", 2, attributes={"phase": phase})
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
        ("tis/batch_skipped_no_logprobs", 0),
        ("tis/skipped_fraction", 0),
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
            ("policy/grad_cosine", 0.2),
            ("policy/grad_cosine_min", -0.3),
            ("policy/grad_cosine_max", 0.4),
            ("policy/grad_norm_reduced", 3),
            ("policy/offpolicy_mask/masked_fraction", 0.05),
            ("policy/offpolicy_mask/vetoed_sequence_fraction", 0.02),
            ("policy/m2_mask/m2_before", 0.03),
            ("policy/m2_mask/masked_fraction", 0.07),
            ("policy/tis/imp_ratio_capped_fraction", 0.12),
            ("policy/ppo_clip_ratio", 0.08),
        ]:
            add(
                "training_metric_value",
                value * scale,
                attributes={"metric": metric, "payload_kind": "train", "step": str(step)},
            )
    add("telemetry_lost_records", 0)
    add("telemetry_rejected_records", 0)
    # One copy of every row per predicate the panels filter on, each copy falsifying one
    # predicate and carrying a value no assertion below expects. The other run is a sync
    # run, which the run picker has to leave out of its dropdown as well.
    sync_resource = json.dumps({"role": "trainer", "host": "trainer", "training_type": "sync"})
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
    seed(database, rows + distractors)
    return database


# Rows each panel returns from the fixture. A GROUP BY that drops a dimension, a fan-out
# that collapses, or a join that duplicates its left side still returns rows, just not this many.
PANEL_ROWS = {
    "Optimizer and synced policy steps": 2,
    "Response tokens generated and trained on / s": 3,
    "Process lifecycle": 1,
    "Driver step, preparation and policy walls": 5,
    "Generation worker await duration": 2,
    "Completed buffer depth and capacity": 2,
    "Buffer dwell of trained groups": 3,
    "Policy staleness at training": 3,
    "Successful rollout-call latency": 3,
    "Weight sync and policy offload walls": 3,
    "Driver event-loop lag": 1,
    "Signed timing residuals": 2,
    "Rollouts completing during policy training": 1,
    "Group dispositions / bucket": 2,
    "Tokens by group disposition / bucket": 2,
    "Training reward and informative groups": 1,
    "Evaluation scores": 8,
    "Length stops and coverage in trained groups": 2,
    "Optimizer diagnostics": 2,
    "Megatron policy wall by rank": 1,
    "Megatron phase detail": 2,
    "Exporter and nonfinite observations": 2,
    "Pre-update model log-ratio drift": 2,
    "Pre-update PPO-window pressure": 2,
    "Drift coverage and token-weight concentration": 2,
    "Mean squared model log-ratio": 1,
    "TIS correction skips": 2,
    "Core and cycle duration": 2,
    "Loss tokens trained per second": 2,
    "Core wall fractions": 1,
    "Useful tokens per configured role GPU-second": 1,
    "Configured role GPU counts": 2,
    "Learner memory by phase": 1,
    "Inference sampled throughput by learner phase": 4,
    "Cumulative core GPU-hours in selected window": 1,
    "Uniform-staleness batch diagnostics": 1,
    "Uniform-staleness diagnostic token coverage": 1,
    "Evaluation response length and stop coverage": 5,
    "Evaluation score contributions by stop class": 3,
    "Staleness of trained groups: groups per step": 7,
    "Staleness of trained groups: tokens per step": 4,
    "Weight-sync stages": 3,
    "Trainer/vLLM logprob mismatch \u03c1 (staleness 0)": 6,
    "Trainer/vLLM logprob mismatch by staleness bucket": 4,
    "Trainer logprob drift within the update": 4,
    "Position dependence of |log \u03c1|": 8,
    "Gradient direction persistence": 9,
    "Correction activity": 14,
}


@pytest.mark.parametrize("title", PANELS)
def test_every_panel_view_returns_declared_fields_for_selected_attempt(store, title):
    assert len(query(store, title)) == PANEL_ROWS[title]


def _bridge(database, *, max_rows=1000):
    queries = []

    def query_source(sql, *, max_rows):
        queries.append(sql)
        return database.execute(sql).fetch_arrow_table()

    source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query_source)
    return create_app(replace(bridge_config(), max_rows=max_rows), {"marin": source}, {}, None, None, None), queries


REQUEST = {
    "clusters": CLUSTER,
    "run": RUN_ID,
    "job": JOB_ID,
    "executions": f"{DRIVER},{WORKER}",
    "from": WINDOW_START_MS,
    "to": WINDOW_START_MS + WINDOW_MS,
}


def test_the_page_reads_finelog_once_per_source_for_every_panel(store):
    app, queries = _bridge(store)

    with TestClient(app) as client:
        responses = {
            title: client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": panel_view(title)})
            for title in PANELS
        }

    assert len(queries) == len(dataset().sources)
    assert {response.status_code for response in responses.values()} == {200}


def test_a_request_past_the_budget_asks_the_operator_to_narrow_it(store):
    app, _ = _bridge(store, max_rows=3)

    with TestClient(app) as client:
        capped = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "policy_step"})
        too_wide = client.get(
            f"/finelog/marin{ENDPOINT}",
            params={**REQUEST, "from": WINDOW_START_MS - 8 * 24 * 3600 * 1000, "view": "policy_step"},
        )

    assert capped.status_code == 400
    assert capped.json()["error"].endswith("narrow the async RL overview filters or time range")
    assert too_wide.status_code == 400
    assert too_wide.json() == {"error": "async RL overview range must not exceed 7 days"}


def test_the_seeded_row_matches_the_table_it_is_inserted_into():
    # seed() inserts positionally, so a column added to one and not the other would shift every
    # field after it onto the wrong column instead of failing.
    assert TelemetryRow._fields == tuple(SCHEMA.names)


def test_wait_means_use_await_counts_and_queue_gauges_use_last_value(store):
    waits = {row["series"]: row["value"] for row in query(store, "Generation worker await duration")}
    assert waits == {"slot mean · driver": pytest.approx(40 / 12), "slot max · driver": 18}
    gauges = {row["series"]: row["value"] for row in query(store, "Completed buffer depth and capacity")}
    assert gauges == {"rollout_queue_depth · driver": 3, "rollout_capacity · driver": 64}


def test_drift_panels_preserve_signed_values_and_do_not_invent_missing_observations(store):
    drift = {row["series"]: row["value"] for row in query(store, "Pre-update model log-ratio drift")}
    assert drift == {
        "policy/mismatch/pooled/log_ratio_mean · driver": -0.1,
        "policy/mismatch/pooled/log_ratio_abs_p99 · driver": 0.7,
    }
    store.execute(f"DELETE FROM {TABLE} WHERE name='training_metric_value'")
    assert query(store, "Drift coverage and token-weight concentration") == []


def test_useful_work_panels_keep_core_and_cycle_denominators_separate(store):
    rates = {row["series"]: row["value"] for row in query(store, "Loss tokens trained per second")}
    assert rates == {
        "async/performance/consumed_loss_tokens_per_core_second · driver": 100,
        "async/performance/consumed_loss_tokens_per_cycle_second · driver": 40,
    }


def test_consumed_length_stops_distinguish_zero_from_incomplete_coverage(store):
    title = "Length stops and coverage in trained groups"
    values = {row["series"]: row["value"] for row in query(store, title)}
    assert values == {
        "consumed/length_stop_fraction · driver": 0.25,
        "consumed/stop_reason_coverage · driver": 1,
    }
    store.execute(f"UPDATE {TABLE} SET value=0 WHERE json_get(attributes_json,'metric')='consumed/length_stop_fraction'")
    assert next(row["value"] for row in query(store, title) if "length_stop_fraction" in row["series"]) == 0
    store.execute(f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric')='consumed/length_stop_fraction'")
    store.execute(
        f"UPDATE {TABLE} SET value=0.5 WHERE json_get(attributes_json,'metric')='consumed/stop_reason_coverage'"
    )
    assert {row["series"]: row["value"] for row in query(store, title)} == {
        "consumed/length_stop_fraction · driver": None,
        "consumed/stop_reason_coverage · driver": 0.5,
    }
    store.execute(f"DELETE FROM {TABLE} WHERE name='training_metric_value'")
    assert query(store, title) == []


def test_overlap_counts_calls_finishing_inside_the_training_window_and_distinguishes_unknown(store):
    title = "Rollouts completing during policy training"
    assert query(store, title) == [
        {"execution_uid": "driver", "step": 1, "coverage": "observed", "completed_calls": 2, "returned_tokens": 18}
    ]
    store.execute(
        f"DELETE FROM {TABLE} WHERE name='rollout_call' "
        f"AND CAST(json_get(body_json,'finished_unix_ms') AS BIGINT)<{BASE_EPOCH_MS + 10000}"
    )
    assert query(store, title)[0]["completed_calls"] == 0
    store.execute(f"DELETE FROM {TABLE} WHERE name='rollout_call'")
    assert query(store, title)[0]["completed_calls"] is None
    assert query(store, title)[0]["returned_tokens"] is None
    assert query(store, title)[0]["coverage"] == "no rollout records"


def test_window_clipped_training_window_reports_unknown_overlap(store):
    store.execute(
        f"UPDATE {TABLE} SET body_json=json_merge_patch(body_json, "
        f"'{{\"started_unix_ms\": {WINDOW_START_MS - 1000}}}') "
        "WHERE name='async_phase_window' AND json_get(attributes_json,'phase')='training'"
    )
    store.execute(
        f"UPDATE {TABLE} SET timestamp_ms={WINDOW_START_MS - 1000} WHERE name='rollout_call' "
        f"AND CAST(json_get(body_json,'finished_unix_ms') AS BIGINT)<{BASE_EPOCH_MS + 10000}"
    )
    row = query(store, "Rollouts completing during policy training")[0]
    assert row["coverage"] == "partial interval"
    assert row["completed_calls"] is None
    assert row["returned_tokens"] is None


def test_health_sums_nonfinite_deltas_and_keeps_exporter_processes_separate(store):
    for name, value, process in [
        ("training_nonfinite_values", 1, "trainer"),
        ("training_nonfinite_values", 1, "trainer"),
        ("telemetry_lost_records", 2, "trainer"),
        ("telemetry_lost_records", 3, "trainer"),
        ("telemetry_lost_records", 4, "other"),
    ]:
        # The panel keys an exporter process by its whole resource attribute set, so these rows
        # carry the fixture's attributes: a second "trainer/trainer" group would tie with the
        # fixture's under ORDER BY and make the dict below keep whichever sorted last.
        store.execute(
            f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            telemetry_row(
                name,
                value,
                timestamp=BASE_EPOCH_MS + 1000,
                seq=1000,
                resource={"role": "trainer", "host": process, "training_type": "async"},
            ),
        )
    rows = query(store, "Exporter and nonfinite observations")
    values = {(row["process"], row["name"]): row["observed_value"] for row in rows}
    assert len(rows) == len(values)
    assert values == {
        ("trainer/trainer", "training_nonfinite_values"): 2,
        ("trainer/trainer", "telemetry_lost_records"): 3,
        ("other/trainer", "telemetry_lost_records"): 4,
        ("trainer/trainer", "telemetry_rejected_records"): 0,
    }


def test_native_work_and_residuals_are_not_clamped_or_merged_across_attempts(store):
    rates = {row["series"]: row["value"] for row in query(store, "Response tokens generated and trained on / s")}
    assert rates == {
        "generated_token · driver": 0.5,
        "consumed_response_token · driver": pytest.approx(1 / 3),
        "consumed_loss_token · driver": 0.3,
    }
    residuals = query(store, "Signed timing residuals")
    assert sorted(row["value"] for row in residuals) == [-0.5, -0.25]
    assert query(store, "Training reward and informative groups")[0]["value"] == 0.5


def test_empty_telemetry_is_unknown_and_startup_only_runs_are_discoverable(store):
    store.execute(f"DELETE FROM {TABLE} WHERE name NOT IN ('lifecycle','terminal')")
    assert store.execute(resolve(run_variable_sql())).fetchall() == [("run",)]
    assert query(store, "Response tokens generated and trained on / s") == []
    assert query(store, "Rollouts completing during policy training") == []


def test_run_picker_offers_the_asynchronous_run_and_not_the_synchronous_one(store):
    # Synchronous runs reach the same table under their own run ids; this dashboard charts
    # only the asynchronous loop, so its picker must not offer one.
    assert store.execute(resolve(run_variable_sql())).fetchall() == [("run",)]


def test_phase_service_rates_exclude_resets_boundaries_and_other_collectors(store):
    rows = query(store, "Inference sampled throughput by learner phase")
    selected = {(row["phase"], row["engine"]): row for row in rows}
    assert len(rows) == 4
    a = selected["training", "engine-A"]
    assert a["tokens_per_sampled_second"] == 10
    assert a["coverage_fraction"] == 0.6
    assert a["intervals"] == 2
    assert a["phase_seconds"] == 10
    b = selected["training", "engine-B"]
    assert b["tokens_per_sampled_second"] == 20
    assert b["coverage_fraction"] == 0.4
    for engine in ["engine-A", "engine-B"]:
        weight_sync = selected["weight_sync", engine]
        assert weight_sync["tokens_per_sampled_second"] is None
        assert weight_sync["coverage_fraction"] == 0
        assert weight_sync["intervals"] == 0


@pytest.mark.parametrize(
    "title", ["Buffer dwell of trained groups", "Policy staleness at training", "Successful rollout-call latency"]
)
def test_percentile_panels_separate_median_tail_and_worst_observation(store, title):
    # DuckDB's quantile_cont and finelog's t-digest disagree on the exact number, so the
    # contract here is the fan-out: three series, ordered, none collapsed onto another.
    values = {row["series"].split(" \u00b7 ")[0]: row["value"] for row in query(store, title)}
    assert set(values) == {"p50", "p95", "max"}
    assert values["p50"] < values["p95"] < values["max"]


def test_learner_memory_keeps_interval_peak_separate_from_current_and_device_usage(store):
    rows = query(store, "Learner memory by phase")
    assert len(rows) == 1
    row = rows[0]
    assert row["gpu"] == "GPU-A"
    assert row["peak_allocated_gib"] == 4
    assert row["peak_reserved_gib"] == 6
    assert row["sampled_allocated_gib"] == 3
    assert row["sampled_free_gib"] == 1
    assert row["device_total_gib"] == 8


def test_core_gpu_hours_charge_both_roles_and_require_complete_counts(store):
    title = "Cumulative core GPU-hours in selected window"
    assert query(store, title)[0]["value"] == pytest.approx(10 * 16 / 3600)
    store.execute(
        f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric')='async/performance/configured_inference_gpus'"
    )
    assert query(store, title) == []


def test_phase_service_rates_reject_clock_adjusted_windows(store):
    store.execute(
        f"UPDATE {TABLE} "
        "SET body_json=json_merge_patch(body_json,'{\"duration_seconds\":99}') WHERE name='async_phase_window'"
    )
    assert query(store, "Inference sampled throughput by learner phase") == []


@pytest.fixture
def staleness_store(telemetry_table):
    """A telemetry table holding only the batches each weighted-staleness case seeds."""
    return telemetry_table


STALENESS_METRICS = {
    "staleness_min": "async/staleness_min",
    "staleness_max": "async/staleness_max",
    "loss": "async/performance/consumed_loss_tokens",
    "tokens": "async/performance/consumed_response_tokens",
    "seqs": "consumed/sequences",
    "mslr": "policy/mismatch/pooled/log_ratio_mean_squared",
    "ess": "policy/mismatch/pooled/ess_fraction",
    "reward": "reward/avg_raw_reward",
    "finite": "policy/mismatch/pooled/finite_fraction",
    "missing": "policy/mismatch/pooled/missing_behavior",
    "policy_loss": "policy/policy_loss",
}


def add_staleness_batch(database, step, *, omit=None, job=JOB_ID, execution=DRIVER, phase="train", **changes):
    values = dict(
        staleness_min=1,
        staleness_max=1,
        loss=100,
        tokens=200,
        seqs=2,
        mslr=0.1,
        ess=0.9,
        reward=0.25,
        finite=1,
        missing=0,
        policy_loss=-0.01,
    )
    values.update(changes)
    rows = [
        telemetry_row(
            "training_metric_value",
            value,
            job=job,
            execution=execution,
            seq=step,
            attributes={"step": str(step), "metric": STALENESS_METRICS[key], "payload_kind": phase},
        )
        for key, value in values.items()
        if key != omit
    ]
    seed(database, rows)


def query_staleness_panels(database):
    return (
        query(database, "Uniform-staleness batch diagnostics"),
        query(database, "Uniform-staleness diagnostic token coverage")[0],
    )


def test_weighting_mixed_duplicates_and_filters(staleness_store):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 1)  # duplicate delivery must not double count
    add_staleness_batch(staleness_store, 2, loss=300, tokens=1200, seqs=3, mslr=0.3, ess=0.8, reward=0.75)
    add_staleness_batch(
        staleness_store, 3, staleness_min=0, staleness_max=2, loss=600, mslr=900
    )  # integer mean still mixed
    add_staleness_batch(staleness_store, 4, job="other-job", loss=10000)
    add_staleness_batch(staleness_store, 4, execution="other-execution", loss=10000)
    add_staleness_batch(staleness_store, 4, phase="eval", loss=10000)
    table, coverage = query_staleness_panels(staleness_store)
    assert len(table) == 1
    row = table[0]
    assert row["staleness"] == 1 and row["updates"] == 2
    assert row["mean_response_tokens"] == 280
    assert row["token_weighted_mslr"] == pytest.approx(0.25)
    assert row["minimum_ess_fraction"] == 0.8 and row["mean_update_raw_reward"] == 0.5
    assert (
        coverage["uniform_staleness_token_fraction"] == 0.4
        and coverage["mixed_staleness_loss_tokens"] == 600
        and coverage["observed_loss_tokens"] == 1000
    )
    assert coverage["uniform_updates"] == 2 and coverage["excluded_updates"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"staleness_min": 0.5, "staleness_max": 0.5},
        {"staleness_min": -1, "staleness_max": -1},
        {"omit": "staleness_min"},
        {"omit": "mslr"},
        {"finite": 0.9},
        {"missing": 1},
        {"mslr": float("nan")},
        {"mslr": float("inf")},
        {"ess": 0},
        {"seqs": float("inf")},
        {"seqs": float("nan")},
        {"seqs": 0.5},
        {"tokens": float("inf")},
        {"tokens": float("nan")},
        {"tokens": 0.5},
    ],
)
def test_incomplete_or_invalid_batch_keeps_token_denominator(staleness_store, change):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 2, loss=300, **change)
    table, coverage = query_staleness_panels(staleness_store)
    assert table[0]["updates"] == 1
    assert coverage["uniform_staleness_token_fraction"] == 0.25 and coverage["observed_loss_tokens"] == 400


@pytest.mark.parametrize("loss", [None, -1, 0.5, float("inf"), float("nan")])
def test_invalid_loss_denominator_is_unavailable(staleness_store, loss):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 2, loss=loss)
    _, coverage = query_staleness_panels(staleness_store)
    assert coverage["uniform_staleness_token_fraction"] is None and coverage["observed_loss_tokens"] is None


def test_conflicting_diagnostic_excluded_but_conflicting_loss_invalidates_coverage(staleness_store):
    add_staleness_batch(staleness_store, 1)
    add_staleness_batch(staleness_store, 1, mslr=0.2)
    table, coverage = query_staleness_panels(staleness_store)
    assert (
        table[0]["staleness"] is None
        and coverage["uniform_staleness_token_fraction"] == 0
        and coverage["observed_loss_tokens"] == 100
    )
    add_staleness_batch(staleness_store, 1, loss=101)
    _, coverage = query_staleness_panels(staleness_store)
    assert coverage["uniform_staleness_token_fraction"] is None


def test_empty_selection_is_explicit_and_unavailable(staleness_store):
    table, coverage = query_staleness_panels(staleness_store)
    assert table[0]["status"] == "No qualifying uniform-staleness batches" and table[0]["staleness"] is None
    assert coverage["uniform_staleness_token_fraction"] is None and coverage["observed_loss_tokens"] is None


def test_evaluation_stop_panels_preserve_score_contributions_and_missing_coverage(store):
    length = {row["series"]: row["value"] for row in query(store, "Evaluation response length and stop coverage")}
    assert length == {
        "eval/all/response_tokens_mean · driver": 150,
        "eval/all/response_tokens_max · driver": 300,
        "eval/all/length_stop_fraction · driver": 0.5,
        "eval/all/completed_stop_fraction · driver": 0.5,
        "eval/all/stop_reason_coverage · driver": 1,
    }
    score = {row["series"]: row["value"] for row in query(store, "Evaluation score contributions by stop class")}
    assert score == {
        "eval/all/avg_score · driver": 0.25,
        "eval/all/length_stop_score_contribution · driver": 0.2,
        "eval/all/completed_stop_score_contribution · driver": 0.05,
    }
    store.execute(
        f"DELETE FROM {TABLE} WHERE "
        "json_get(attributes_json,'metric') LIKE 'eval/%/length_stop_%' OR "
        "json_get(attributes_json,'metric') LIKE 'eval/%/completed_stop_%'"
    )
    # Partial/legacy coverage has no fraction/contribution records: the SQL must not synthesize zeros.
    length = query(store, "Evaluation response length and stop coverage")
    assert len(length) == 3
    assert query(store, "Evaluation score contributions by stop class")[0]["value"] == 0.25
    assert len(query(store, "Evaluation score contributions by stop class")) == 1
    store.execute(f"DELETE FROM {TABLE}")
    assert query(store, "Evaluation response length and stop coverage") == []
    assert query(store, "Evaluation score contributions by stop class") == []


def test_periodic_evaluation_metrics_logged_in_train_phase_are_visible(store):
    # The real trainer logs the initial eval separately, then merges periodic evals into its training row.
    store.execute(
        f"UPDATE {TABLE} SET "
        'attributes_json=json_merge_patch(attributes_json, \'{"payload_kind":"train"}\') '
        "WHERE json_get(attributes_json,'metric') LIKE 'eval/%'"
    )
    assert len(query(store, "Evaluation response length and stop coverage")) == 5
    assert len(query(store, "Evaluation score contributions by stop class")) == 3


def test_consumed_staleness_panels_count_groups_and_sum_tokens_per_staleness(store):
    groups = query(store, "Staleness of trained groups: groups per step")
    tokens = query(store, "Staleness of trained groups: tokens per step")
    assert [row["value"] for row in groups if row["series"].startswith("staleness 0 ·")] == [1, 1, 1]
    assert [row["value"] for row in groups if row["series"].startswith("staleness 1 ·")] == [1, 2, 1]
    assert [row["value"] for row in tokens if row["series"].startswith("staleness 1 ·")] == [80, 70]
    assert [row["value"] for row in tokens if row["series"].startswith("staleness 0 ·")] == [10, 20]
    assert tokens[0]["t"] == tokens[1]["t"] and tokens[2]["t"] == tokens[3]["t"]
    store.execute(
        f"DELETE FROM {TABLE} WHERE "
        "(name='rollout_staleness_steps' AND value<>0) OR "
        "(name='consumed_staleness' AND json_get(body_json,'staleness')='1')"
    )
    for title in ("Staleness of trained groups: groups per step", "Staleness of trained groups: tokens per step"):
        assert all(row["series"].startswith("staleness 0 ·") for row in query(store, title))
    store.execute(f"DELETE FROM {TABLE} WHERE name='consumed_staleness'")
    assert query(store, "Staleness of trained groups: tokens per step") == []


def test_weight_sync_timeline_orders_training_and_sync_windows(store):
    rows = query(store, "Weight-sync stages")
    assert {row["execution"] for row in rows} == {"driver"}
    spans = {row["state"]: row for row in rows}
    assert set(spans) == {"training", "weight sync", "weights synced"}
    assert spans["training"]["finish"] == spans["weight sync"]["start"]
    assert spans["training"]["finish"] - spans["training"]["start"] == 10_000
    assert spans["weight sync"]["finish"] - spans["weight sync"]["start"] == 4_000
    assert spans["weights synced"]["finish"] - spans["weights synced"]["start"] == 1


def test_ratio_panels_read_mismatch_and_learner_drift_families_separately(store):
    mismatch = query(store, "Trainer/vLLM logprob mismatch \u03c1 (staleness 0)")
    staleness = query(store, "Trainer/vLLM logprob mismatch by staleness bucket")
    drift = query(store, "Trainer logprob drift within the update")
    assert [row["value"] for row in mismatch if "/log_ratio_abs_mean ·" in row["series"]] == [0.1, 0.2]
    assert [row["value"] for row in staleness if "/staleness1/" in row["series"]] == [0.3, 0.6]
    assert [row["value"] for row in drift if row["series"].startswith("policy/log_ratio_abs_mean ·")] == [0.05, 0.1]
    assert not any("mismatch" in row["series"] for row in drift)
    store.execute(f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric') LIKE 'policy/mismatch/staleness1/%'")
    assert not any(
        "/staleness1/" in row["series"] for row in query(store, "Trainer/vLLM logprob mismatch by staleness bucket")
    )


def test_position_panel_keeps_ratio_families_and_positions_separate(store):
    rows = query(store, "Position dependence of |log \u03c1|")
    assert [row["value"] for row in rows if "mismatch/pooled/pos_last256" in row["series"]] == [0.23, 0.46]
    assert [row["value"] for row in rows if "log_ratio_pos_last256" in row["series"]] == [0.04, 0.08]
    assert not any("pos_middle" in row["series"] for row in rows)


def test_gradient_direction_panel_reports_band_and_reduced_norm(store):
    rows = query(store, "Gradient direction persistence")
    assert [row["value"] for row in rows if "grad_cosine_min" in row["series"]] == [-0.3, -0.6]
    assert [row["value"] for row in rows if "grad_cosine_max" in row["series"]] == [0.4, 0.8]
    assert [row["value"] for row in rows if "grad_norm_reduced" in row["series"]] == [3, 6]
    assert [row["value"] for row in rows if "raw_grad_norm" in row["series"]] == [4]


def test_correction_panel_distinguishes_populations_and_bounds_reference_coverage(store):
    rows = query(store, "Correction activity")
    assert [row["value"] for row in rows if "vetoed_sequence_fraction" in row["series"]] == [0.02, 0.04]
    assert [row["value"] for row in rows if "policy/tis/imp_ratio_capped_fraction" in row["series"]] == [0.12, 0.24]
    assert [row["value"] for row in rows if "offpolicy_mask/masked_fraction" in row["series"]] == [0.05, 0.1]
    assert [row["value"] for row in rows if row["series"].startswith("M2 reference")] == [0.04, 0.04]
    store.execute(f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric')='policy/m2_mask/m2_before'")
    assert not any(row["series"].startswith("M2 reference") for row in query(store, "Correction activity"))
