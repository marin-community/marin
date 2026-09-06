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

        def add(name, value=0, *, attributes=None, body=None, process="trainer", execution="driver", timestamp=None):
            rows.append(
                (
                    "cw-us-east-02a",
                    "marinskyrl",
                    "run",
                    "job",
                    execution,
                    1788566460000 + len(rows) if timestamp is None else timestamp,
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
            ("policy/behavior_drift/log_ratio_mean", -0.1),
            ("policy/behavior_drift/mean_squared_log_ratio", 0.04),
            ("tis/batch_skipped_no_logprobs", 0),
            ("tis/skipped_fraction", 0),
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
            ("async/performance/configured_inference_gpus", 8),
        ]:
            add(
                "training_metric_value",
                value,
                attributes={"metric": metric, "phase": "eval" if metric.startswith("eval/") else "train"},
            )
        add(
            "cuda_memory_observation",
            execution="worker",
            process="learner",
            attributes={
                "worker_role": "policy",
                "rank": "0",
                "gpu_uuid": "GPU-A",
                "phase": "ppo_forward_backward_update",
            },
            body={
                "peak_allocated_bytes": 4 * 2**30,
                "peak_reserved_bytes": 6 * 2**30,
                "allocated_bytes": 3 * 2**30,
                "device_free_bytes": 2**30,
                "device_total_bytes": 8 * 2**30,
            },
        )
        phase_start = 1788566460000
        for phase, start, finish in [("training", 0, 10000), ("publication", 10000, 14000)]:
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
        # 8->12s crosses publication boundary: cannot attribute it to either phase.
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
        add(
            "weight_change_probe",
            body={
                "target_update": 2,
                "status": "valid",
                "coverage_complete": 1,
                "dense_wire_bytes": 1000,
                "estimated_changed_element_fraction": 0.025,
                "estimated_index32_value_bytes": 75,
                "capture_enqueue_seconds": 0.01,
                "sample_cuda_milliseconds": 2,
                "compare_commit_seconds": 0.03,
            },
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


def test_consumed_length_stops_distinguish_zero_from_incomplete_coverage(store):
    title = "Consumed length stops and coverage"
    values = {row["series"]: row["value"] for row in query(store, title)}
    assert values == {
        "consumed/length_stop_fraction · driver": 0.25,
        "consumed/stop_reason_coverage · driver": 1,
    }
    store.execute(
        'UPDATE "telemetry_v1.marinskyrl" SET value=0 '
        "WHERE json_get(attributes_json,'metric')='consumed/length_stop_fraction'"
    )
    assert next(row["value"] for row in query(store, title) if "length_stop_fraction" in row["series"]) == 0
    store.execute(
        'DELETE FROM "telemetry_v1.marinskyrl" '
        "WHERE json_get(attributes_json,'metric')='consumed/length_stop_fraction'"
    )
    store.execute(
        'UPDATE "telemetry_v1.marinskyrl" SET value=0.5 '
        "WHERE json_get(attributes_json,'metric')='consumed/stop_reason_coverage'"
    )
    assert {row["series"]: row["value"] for row in query(store, title)} == {
        "consumed/length_stop_fraction · driver": None,
        "consumed/stop_reason_coverage · driver": 0.5,
    }
    store.execute("DELETE FROM \"telemetry_v1.marinskyrl\" WHERE name='training_metric_value'")
    assert query(store, title) == []


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
        publication = selected["publication", engine]
        assert publication["tokens_per_sampled_second"] is None
        assert publication["coverage_fraction"] == 0
        assert publication["intervals"] == 0


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
        'DELETE FROM "telemetry_v1.marinskyrl" '
        "WHERE json_get(attributes_json,'metric')='async/performance/configured_inference_gpus'"
    )
    assert query(store, title) == []


def test_phase_service_rates_reject_clock_adjusted_windows(store):
    store.execute(
        'UPDATE "telemetry_v1.marinskyrl" '
        "SET body_json=json_merge_patch(body_json,'{\"duration_seconds\":99}') WHERE name='async_phase_window'"
    )
    assert query(store, "Inference sampled throughput by learner phase") == []


@pytest.fixture
def age_store(store):
    """An empty native telemetry table for independent weighted-batch cases."""
    store.execute('DELETE FROM "telemetry_v1.marinskyrl"')
    return store


AGE_METRICS = {
    "age_min": "async/staleness_min",
    "age_max": "async/staleness_max",
    "loss": "async/performance/consumed_loss_tokens",
    "tokens": "async/performance/consumed_response_tokens",
    "seqs": "consumed/sequences",
    "mslr": "policy/behavior_drift/mean_squared_log_ratio",
    "ess": "policy/behavior_drift/token_weight_ess_fraction",
    "reward": "reward/avg_raw_reward",
    "finite": "policy/behavior_drift/finite_fraction",
    "missing": "policy/behavior_drift/missing_behavior",
    "policy_loss": "policy/policy_loss",
}


def add_age_batch(d, step, *, omit=None, job="job", execution="driver", phase="train", **changes):
    vals = dict(
        age_min=1,
        age_max=1,
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
    vals.update(changes)
    rows = [
        (
            "cw-us-east-02a",
            "marinskyrl",
            "run",
            job,
            execution,
            1788566460000,
            step,
            "training_metric_value",
            v,
            json.dumps({"step": str(step), "metric": AGE_METRICS[k], "phase": phase}),
            "{}",
            "{}",
        )
        for k, v in vals.items()
        if k != omit
    ]
    d.executemany('INSERT INTO "telemetry_v1.marinskyrl" VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)


def query_age_panels(d):
    return query(d, "Uniform-age batch diagnostics"), query(d, "Uniform-age diagnostic token coverage")[0]


def test_weighting_mixed_duplicates_and_filters(age_store):
    add_age_batch(age_store, 1)
    add_age_batch(age_store, 1)  # duplicate delivery must not double count
    add_age_batch(age_store, 2, loss=300, tokens=1200, seqs=3, mslr=0.3, ess=0.8, reward=0.75)
    add_age_batch(age_store, 3, age_min=0, age_max=2, loss=600, mslr=900)  # integer mean still mixed
    add_age_batch(age_store, 4, job="other-job", loss=10000)
    add_age_batch(age_store, 4, execution="other-execution", loss=10000)
    add_age_batch(age_store, 4, phase="eval", loss=10000)
    table, c = query_age_panels(age_store)
    assert len(table) == 1
    row = table[0]
    assert row["age"] == 1 and row["updates"] == 2
    assert row["mean_response_tokens"] == 280
    assert row["token_weighted_mslr"] == pytest.approx(0.25)
    assert row["minimum_ess_fraction"] == 0.8 and row["mean_update_raw_reward"] == 0.5
    assert (
        c["uniform_age_token_fraction"] == 0.4
        and c["mixed_age_loss_tokens"] == 600
        and c["observed_loss_tokens"] == 1000
    )
    assert c["uniform_updates"] == 2 and c["excluded_updates"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"age_min": 0.5, "age_max": 0.5},
        {"age_min": -1, "age_max": -1},
        {"omit": "age_min"},
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
def test_incomplete_or_invalid_batch_keeps_token_denominator(age_store, change):
    add_age_batch(age_store, 1)
    add_age_batch(age_store, 2, loss=300, **change)
    table, c = query_age_panels(age_store)
    assert table[0]["updates"] == 1
    assert c["uniform_age_token_fraction"] == 0.25 and c["observed_loss_tokens"] == 400


@pytest.mark.parametrize("loss", [None, -1, 0.5, float("inf"), float("nan")])
def test_invalid_loss_denominator_is_unavailable(age_store, loss):
    add_age_batch(age_store, 1)
    add_age_batch(age_store, 2, loss=loss)
    _, c = query_age_panels(age_store)
    assert c["uniform_age_token_fraction"] is None and c["observed_loss_tokens"] is None


def test_conflicting_diagnostic_excluded_but_conflicting_loss_invalidates_coverage(age_store):
    add_age_batch(age_store, 1)
    add_age_batch(age_store, 1, mslr=0.2)
    table, c = query_age_panels(age_store)
    assert table[0]["age"] is None and c["uniform_age_token_fraction"] == 0 and c["observed_loss_tokens"] == 100
    add_age_batch(age_store, 1, loss=101)
    _, c = query_age_panels(age_store)
    assert c["uniform_age_token_fraction"] is None


def test_empty_selection_is_explicit_and_unavailable(age_store):
    table, c = query_age_panels(age_store)
    assert table[0]["status"] == "No qualifying uniform-age batches" and table[0]["age"] is None
    assert c["uniform_age_token_fraction"] is None and c["observed_loss_tokens"] is None


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
        'DELETE FROM "telemetry_v1.marinskyrl" WHERE '
        "json_get(attributes_json,'metric') LIKE 'eval/%/length_stop_%' OR "
        "json_get(attributes_json,'metric') LIKE 'eval/%/completed_stop_%'"
    )
    # Partial/legacy coverage has no fraction/contribution records: the SQL must not synthesize zeros.
    length = query(store, "Evaluation response length and stop coverage")
    assert len(length) == 3
    assert query(store, "Evaluation score contributions by stop class")[0]["value"] == 0.25
    assert len(query(store, "Evaluation score contributions by stop class")) == 1
    store.execute('DELETE FROM "telemetry_v1.marinskyrl"')
    assert query(store, "Evaluation response length and stop coverage") == []
    assert query(store, "Evaluation score contributions by stop class") == []


def test_periodic_evaluation_metrics_logged_in_train_phase_are_visible(store):
    # The real trainer logs the initial eval separately, then merges periodic evals into its training row.
    store.execute(
        'UPDATE "telemetry_v1.marinskyrl" SET '
        'attributes_json=json_merge_patch(attributes_json, \'{"phase":"train"}\') '
        "WHERE json_get(attributes_json,'metric') LIKE 'eval/%'"
    )
    assert len(query(store, "Evaluation response length and stop coverage")) == 5
    assert len(query(store, "Evaluation score contributions by stop class")) == 3
