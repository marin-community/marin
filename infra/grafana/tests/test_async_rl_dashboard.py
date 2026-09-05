# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the shipped async dashboard SQL against adversarial telemetry rows."""

import json
from pathlib import Path

import duckdb
import pytest

DASHBOARD = json.loads((Path(__file__).parents[1] / "dashboards/async_rl.json").read_text())
PANELS = {panel["title"]: panel for panel in DASHBOARD["panels"] if "targets" in panel}


def resolve(sql):
    for macro, value in {
        "{{from}}": "TIMESTAMP '2026-09-05 00:00:00'",
        "{{to}}": "TIMESTAMP '2026-09-05 00:05:00'",
        "${__interval_ms}": "300000",
        "${cluster:sqlstring}": "'cw-us-east-02a'",
        "${run:sqlstring}": "'run'",
        "${job:sqlstring}": "'job'",
        "${execution:sqlstring}": "'driver','worker'",
    }.items():
        sql = sql.replace(macro, value)
    return sql


def query(database, title):
    target = PANELS[title]["targets"][0]
    sql = next(param["value"] for param in target["url_options"]["params"] if param["key"] == "sql")
    cursor = database.execute(resolve(sql))
    columns = [column[0] for column in cursor.description]
    assert columns == [column["selector"] for column in target["columns"]]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


@pytest.fixture
def store():
    with duckdb.connect() as database:
        database.execute(
            'CREATE TABLE "telemetry_v1.marinskyrl" ("cluster" VARCHAR,service VARCHAR,run_id VARCHAR,'
            "job_id VARCHAR,execution_uid VARCHAR,timestamp_ms BIGINT,seq BIGINT,name VARCHAR,value DOUBLE,"
            "attributes_json VARCHAR,resource_attributes_json VARCHAR,body_json VARCHAR)"
        )
        database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)::TIMESTAMP")
        database.execute("CREATE MACRO date_bin(width, moment) AS time_bucket(width, moment)")
        database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
        database.execute("CREATE MACRO approx_percentile_cont(value, q) AS quantile_cont(value, q)")
        rows = []

        def add(name, value=0, *, attributes=None, body=None, process="trainer", execution="driver"):
            rows.append(
                (
                    "cw-us-east-02a",
                    "marinskyrl",
                    "run",
                    "job",
                    execution,
                    1788566460000 + len(rows),
                    len(rows),
                    name,
                    value,
                    json.dumps({"role": "trainer", "step": "1", **(attributes or {})}),
                    json.dumps({"role": "trainer", "host": process}),
                    json.dumps(body or {}),
                )
            )

        add("lifecycle", body={"state": "started"})
        add("terminal", body={"status": "completed", "reason": "normal_exit"})
        add("policy_step", 1)
        add("policy_weights_published", body={"completed_update": 1})
        for kind, value in [("generated_token", 150), ("consumed_response_token", 100), ("consumed_loss_token", 90)]:
            add("work_completed", value, attributes={"work_kind": kind})
        for phase in (
            "step",
            "wait_for_generation_buffer",
            "run_training",
            "fwd_logprobs_values_reward",
            "train_critic_and_policy",
            "weight_pause",
            "weight_broadcast",
            "weight_resume",
        ):
            add("phase_duration_seconds", 2, attributes={"phase": phase})
        for value in (10, 30):
            add("rollout_wait_seconds", value, attributes={"wait": "slot", "stat": "sum"})
        for value in (2, 10):
            add("rollout_wait_count", value, attributes={"wait": "slot"})
        add("rollout_wait_seconds", 18, attributes={"wait": "slot", "stat": "max"})
        for value in (1, 5, 3):
            add("rollout_queue_depth", value)
        add("rollout_capacity", 64)
        add("rollout_buffer_dwell_seconds", 0.5, attributes={"outcome": "consumed"})
        add("rollout_staleness_steps", 1)
        add("phase_duration_seconds", 2, attributes={"phase": "rollout_call", "outcome": "success"})
        add("phase_duration_seconds", -0.25, attributes={"phase": "rollout_call_residual", "outcome": "success"})
        add(
            "phase_duration_seconds",
            8,
            attributes={"phase": "megatron_policy_train_total", "outcome": "success", "rank": "0"},
            execution="worker",
        )
        add(
            "phase_duration_seconds",
            -0.5,
            attributes={"phase": "megatron_policy_train_residual", "outcome": "success", "rank": "0"},
            execution="worker",
        )
        add("event_loop_lag_seconds", 0.1)
        add("policy_training_interval", attributes={"outcome": "success"}, body={"started": 10, "finished": 20})
        for call, finish, tokens, process in [
            ("a", 15, 7, "trainer"),
            ("b", 19, 11, "trainer"),
            ("outside", 25, 17, "trainer"),
            ("different-clock", 15, 999, "other"),
        ]:
            add(
                "rollout_call",
                attributes={"outcome": "success"},
                body={"call_id": call, "started": 9, "finished": finish, "response_tokens": tokens},
                process=process,
            )
        for outcome, tokens in [("consumed", 100), ("epoch_discarded", 50)]:
            add("rollout_group_count", 1, attributes={"outcome": outcome})
            add("rollout_group_tokens", tokens, attributes={"outcome": outcome})
        for metric, value in [
            ("reward/avg_raw_reward", 0.5),
            ("eval/all/avg_score", 0.25),
            ("policy/policy_loss", -0.2),
            ("policy/raw_grad_norm", 4),
            ("policy/behavior_drift/log_ratio_mean", -0.1),
            ("policy/behavior_drift/abs_log_ratio_p99", 0.7),
            ("policy/behavior_drift/lower_clip_pressure", 0.2),
            ("policy/behavior_drift/upper_clip_pressure", 0.1),
            ("policy/behavior_drift/finite_fraction", 0.75),
            ("policy/behavior_drift/token_weight_ess_fraction", 0.8),
            ("async/performance/core_seconds", 10),
            ("async/performance/cycle_seconds", 25),
            ("async/performance/consumed_loss_tokens_per_core_second", 100),
            ("async/performance/consumed_loss_tokens_per_cycle_second", 40),
            ("async/performance/buffer_wait_fraction", 0.2),
            ("async/performance/loss_tokens_per_configured_policy_gpu_second", 5),
            ("async/performance/configured_policy_gpus", 8),
        ]:
            add(
                "training_metric_value",
                value,
                attributes={"metric": metric, "phase": "eval" if metric.startswith("eval/") else "train"},
            )
        add("telemetry_lost_records", 0)
        add("telemetry_rejected_records", 0)
        # Same run/step but another job, another execution, or no execution must not contaminate panels.
        distractors = []
        for row in rows:
            for index, replacement in ((3, "other-job"), (4, "other-execution"), (4, None)):
                other = list(row)
                other[index] = replacement
                other[8] = 1000
                distractors.append(tuple(other))
        database.executemany(
            'INSERT INTO "telemetry_v1.marinskyrl" VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows + distractors
        )
        yield database


@pytest.mark.parametrize("title", PANELS)
def test_shipped_panel_sql_returns_declared_fields_for_selected_attempt(store, title):
    assert query(store, title)


def test_wait_means_use_await_counts_and_queue_gauges_use_last_value(store):
    waits = {row["series"]: row["value"] for row in query(store, "Producer await duration")}
    assert waits == {"slot mean · driver": pytest.approx(40 / 12), "slot max · driver": 18}
    gauges = {row["series"]: row["value"] for row in query(store, "Completed buffer depth and capacity")}
    assert gauges == {"rollout_queue_depth · driver": 3, "rollout_capacity · driver": 64}


def test_drift_panels_preserve_signed_values_and_do_not_invent_missing_observations(store):
    drift = {row["series"]: row["value"] for row in query(store, "Pre-update model log-ratio drift")}
    assert drift == {
        "policy/behavior_drift/log_ratio_mean · driver": -0.1,
        "policy/behavior_drift/abs_log_ratio_p99 · driver": 0.7,
    }
    store.execute("DELETE FROM \"telemetry_v1.marinskyrl\" WHERE name='training_metric_value'")
    assert query(store, "Drift coverage and token-weight concentration") == []


def test_useful_work_panels_keep_core_and_cycle_denominators_separate(store):
    rates = {row["series"]: row["value"] for row in query(store, "Consumed loss tokens per second")}
    assert rates == {
        "async/performance/consumed_loss_tokens_per_core_second · driver": 100,
        "async/performance/consumed_loss_tokens_per_cycle_second · driver": 40,
    }


def test_overlap_joins_only_the_identical_process_clock_and_distinguishes_unknown(store):
    title = "Rollouts completing during policy training"
    assert query(store, title) == [
        {"execution_uid": "driver", "step": 1, "coverage": "observed", "completed_calls": 2, "returned_tokens": 18}
    ]
    store.execute(
        "DELETE FROM \"telemetry_v1.marinskyrl\" WHERE name='rollout_call' "
        "AND CAST(json_get(body_json,'finished') AS DOUBLE)<20"
    )
    assert query(store, title)[0]["completed_calls"] == 0
    store.execute("DELETE FROM \"telemetry_v1.marinskyrl\" WHERE name='rollout_call'")
    assert query(store, title)[0]["completed_calls"] is None
    assert query(store, title)[0]["returned_tokens"] is None
    assert query(store, title)[0]["coverage"] == "no rollout records"


def test_window_clipped_policy_interval_reports_unknown_overlap(store):
    store.execute(
        "UPDATE \"telemetry_v1.marinskyrl\" SET timestamp_ms=1788566405000 WHERE name='policy_training_interval'"
    )
    store.execute(
        "UPDATE \"telemetry_v1.marinskyrl\" SET timestamp_ms=1788566399000 WHERE name='rollout_call' "
        "AND CAST(json_get(body_json,'finished') AS DOUBLE)<20"
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
        store.execute(
            'INSERT INTO "telemetry_v1.marinskyrl" VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            [
                "cw-us-east-02a",
                "marinskyrl",
                "run",
                "job",
                "driver",
                1788566461000,
                1000,
                name,
                value,
                "{}",
                json.dumps({"role": "trainer", "host": process}),
                "{}",
            ],
        )
    rows = query(store, "Exporter and nonfinite observations")
    values = {(row["process"], row["name"]): row["observed_value"] for row in rows}
    assert values == {
        ("trainer/trainer", "training_nonfinite_values"): 2,
        ("trainer/trainer", "telemetry_lost_records"): 3,
        ("other/trainer", "telemetry_lost_records"): 4,
        ("trainer/trainer", "telemetry_rejected_records"): 0,
    }


def test_native_work_and_residuals_are_not_clamped_or_merged_across_attempts(store):
    rates = {row["series"]: row["value"] for row in query(store, "Generated and consumed response tokens / s")}
    assert rates == {
        "generated_token · driver": 0.5,
        "consumed_response_token · driver": pytest.approx(1 / 3),
        "consumed_loss_token · driver": 0.3,
    }
    residuals = query(store, "Signed timing residuals")
    assert sorted(row["value"] for row in residuals) == [-0.5, -0.25]
    assert "min" not in PANELS["Signed timing residuals"]["fieldConfig"]["defaults"]
    assert query(store, "Training reward and informative groups")[0]["value"] == 0.5


def test_empty_telemetry_is_unknown_and_startup_only_runs_are_discoverable(store):
    store.execute("DELETE FROM \"telemetry_v1.marinskyrl\" WHERE name NOT IN ('lifecycle','terminal')")
    variable = next(item for item in DASHBOARD["templating"]["list"] if item["name"] == "run")
    params = variable["query"]["infinityQuery"]["url_options"]["params"]
    sql = next(param["value"] for param in params if param["key"] == "sql")
    assert store.execute(resolve(sql)).fetchall() == [("run",)]
    assert query(store, "Generated and consumed response tokens / s") == []
    assert query(store, "Rollouts completing during policy training") == []
