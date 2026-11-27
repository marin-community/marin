# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from rl_execution_fixtures import (
    BASE_EPOCH_MS,
    DRIVER,
    ENDPOINT,
    JOB_ID,
    RUN_ID,
    TABLE,
    WINDOW_MS,
    WINDOW_START_MS,
    WORKER,
    query,
    resolve,
    run_variable_sql,
    seed,
    telemetry_row,
)
from rl_execution_requests import (
    REQUEST,
    _bridge,
)
from starlette.testclient import TestClient


def test_step_composition_closes_on_the_mean_step_and_keeps_executions_separate(telemetry_table):
    rows = []
    for execution, step, duration, children in (
        (DRIVER, 1, 20, (("prepare", 5), ("train", 10))),
        (DRIVER, 2, 40, (("prepare", 10), ("train", 20), ("checkpoint", 5))),
        (WORKER, 1, 100, (("train", 90),)),
    ):
        for phase, value, parent in (("step", duration, ""), *((p, v, "step") for p, v in children)):
            rows.append(
                telemetry_row(
                    "phase_duration_seconds",
                    value,
                    execution=execution,
                    attributes={
                        "step": str(step),
                        "role": "trainer",
                        "clock_domain": "inclusive_wall",
                        "root": "step",
                        "phase": phase,
                        "parent": parent,
                        "outcome": "failure" if step == 2 else "success",
                    },
                )
            )
    seed(telemetry_table, rows)
    app, _ = _bridge(telemetry_table)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "step_composition"})
        coverage = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "span_coverage"})
    assert response.status_code == 200, response.text
    assert response.json() == [
        {"section": "step_composition", "t": BASE_EPOCH_MS, "series": phase + " · " + execution, "value": value}
        for phase, execution, value in (
            ("checkpoint", DRIVER, 2.5),
            ("prepare", DRIVER, 7.5),
            ("step", DRIVER, 5),
            ("step", WORKER, 10),
            ("train", DRIVER, 15),
            ("train", WORKER, 90),
        )
    ]

    assert coverage.status_code == 200
    assert coverage.json() == [
        {"section": "span_coverage", "role": "trainer", "clock_domain": "inclusive_wall", "steps": 3, "failed_steps": 1}
    ]


def test_drift_panels_preserve_signed_values_and_do_not_invent_missing_observations(rl_execution_store):
    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE name='training_metric_value'")
    assert query(rl_execution_store, "Drift coverage and token-weight concentration") == []


def test_consumed_length_stops_distinguish_zero_from_incomplete_coverage(rl_execution_store):
    title = "Length stops and coverage in trained groups"
    values = {row["series"]: row["value"] for row in query(rl_execution_store, title)}
    assert values == {
        "consumed/length_stop_fraction · driver": 0.25,
        "consumed/stop_reason_coverage · driver": 1,
    }
    rl_execution_store.execute(
        f"UPDATE {TABLE} SET value=0 WHERE json_get(attributes_json,'metric')='consumed/length_stop_fraction'"
    )
    assert next(row["value"] for row in query(rl_execution_store, title) if "length_stop_fraction" in row["series"]) == 0
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric')='consumed/length_stop_fraction'"
    )
    rl_execution_store.execute(
        f"UPDATE {TABLE} SET value=0.5 WHERE json_get(attributes_json,'metric')='consumed/stop_reason_coverage'"
    )
    assert {row["series"]: row["value"] for row in query(rl_execution_store, title)} == {
        "consumed/length_stop_fraction · driver": None,
        "consumed/stop_reason_coverage · driver": 0.5,
    }
    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE name='training_metric_value'")
    assert query(rl_execution_store, title) == []


def test_overlap_counts_calls_finishing_inside_the_training_window_and_distinguishes_unknown(rl_execution_store):
    title = "Rollouts completing during policy training"
    assert query(rl_execution_store, title) == [
        {"execution_uid": "driver", "step": 1, "coverage": "observed", "completed_calls": 2, "returned_tokens": 18}
    ]
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE name='rollout_call' "
        f"AND CAST(json_get(body_json,'finished_unix_ms') AS BIGINT)<{BASE_EPOCH_MS + 10000}"
    )
    assert query(rl_execution_store, title)[0]["completed_calls"] == 0
    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE name='rollout_call'")
    assert query(rl_execution_store, title)[0]["completed_calls"] is None
    assert query(rl_execution_store, title)[0]["returned_tokens"] is None
    assert query(rl_execution_store, title)[0]["coverage"] == "no rollout records"

    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE name='async_phase_window' AND json_get(attributes_json,'phase')='training'"
    )
    seed(
        rl_execution_store,
        [
            telemetry_row(
                "async_phase_window",
                1,
                execution=execution,
                timestamp=BASE_EPOCH_MS + finished,
                attributes={"phase": "training", "outcome": "success", "step": str(step)},
                body={"started_unix_ms": BASE_EPOCH_MS + finished - 1000, "finished_unix_ms": BASE_EPOCH_MS + finished},
            )
            for execution, step, finished in [
                *((DRIVER, 100 + i, 20000 + 1000 * i) for i in range(55)),
                (WORKER, 1, 200000),
            ]
        ],
    )
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "training_overlap"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {
                "section": "training_overlap",
                "execution_uid": execution,
                "step": step,
                "coverage": "no rollout records",
                "completed_calls": None,
                "returned_tokens": None,
            }
            for execution, step in [(WORKER, 1), *((DRIVER, step) for step in range(106, 155))]
        ]


def test_window_clipped_training_window_reports_unknown_overlap(rl_execution_store):
    rl_execution_store.execute(
        f"UPDATE {TABLE} SET body_json=json_merge_patch(body_json, "
        f"'{{\"started_unix_ms\": {WINDOW_START_MS - 1000}}}') "
        "WHERE name='async_phase_window' AND json_get(attributes_json,'phase')='training'"
    )
    rl_execution_store.execute(
        f"UPDATE {TABLE} SET timestamp_ms={WINDOW_START_MS - 1000} WHERE name='rollout_call' "
        f"AND CAST(json_get(body_json,'finished_unix_ms') AS BIGINT)<{BASE_EPOCH_MS + 10000}"
    )
    row = query(rl_execution_store, "Rollouts completing during policy training")[0]
    assert row["coverage"] == "partial interval"
    assert row["completed_calls"] is None
    assert row["returned_tokens"] is None


def test_health_sums_nonfinite_deltas_and_keeps_exporter_processes_separate(rl_execution_store):
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
        rl_execution_store.execute(
            f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            telemetry_row(
                name,
                value,
                timestamp=BASE_EPOCH_MS + 1000,
                seq=1000,
                resource={"role": "trainer", "host": process, "training_type": "async"},
            ),
        )
    rows = query(rl_execution_store, "Exporter and nonfinite observations")
    values = {(row["process"], row["name"]): row["observed_value"] for row in rows}
    assert len(rows) == len(values)
    assert values == {
        ("trainer/trainer", "training_nonfinite_values"): 2,
        ("trainer/trainer", "telemetry_lost_records"): 3,
        ("other/trainer", "telemetry_lost_records"): 4,
        ("trainer/trainer", "telemetry_rejected_records"): 0,
    }
    rl_execution_store.execute(
        f"UPDATE {TABLE} SET body_json=? WHERE name='terminal' AND run_id=? AND job_id=? AND execution_uid=?",
        [
            json.dumps(
                {
                    "status": "completed",
                    "reason": "normal_exit",
                    "export_lost_records": "bad",
                    "export_queued_records": 7,
                }
            ),
            RUN_ID,
            JOB_ID,
            DRIVER,
        ],
    )
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "lifecycle"})
    assert response.status_code == 200, response.text
    assert response.json() == [
        {
            "section": "lifecycle",
            "execution_uid": DRIVER,
            "role": "trainer",
            "process": "trainer",
            "event": "terminal",
            "status": "completed",
            "reason": "normal_exit",
            "export_lost_records": None,
            "export_queued_records": 7,
            "last_record_ms": BASE_EPOCH_MS + 1,
        }
    ]


def test_megatron_detail_limits_history_per_rank_without_truncating_wall_summaries(rl_execution_store):
    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE name='phase_duration_seconds'")
    observed = []
    for rank, old_tail, restart in (("0", range(20, 30), range(1, 4)), ("1", range(50, 60), range(1, 3))):
        for execution, steps, offset in ((WORKER, old_tail, 1000), ("restart", restart, 10000)):
            for step in steps:
                observed.append(
                    telemetry_row(
                        "phase_duration_seconds",
                        step + int(rank),
                        execution=execution,
                        timestamp=BASE_EPOCH_MS + offset + step,
                        attributes={
                            "backend": "megatron",
                            "phase": "ppo_train",
                            "outcome": "success",
                            "rank": rank,
                            "step": str(step),
                            "clock_domain": "cpu_dispatch_wall",
                            "role": "worker",
                        },
                    )
                )
    seed(rl_execution_store, observed)
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        detail = client.get(
            f"/finelog/marin{ENDPOINT}", params={**REQUEST, "executions": "__all", "view": "megatron_phases"}
        )
        aggregate = client.get(
            f"/finelog/marin{ENDPOINT}", params={**REQUEST, "executions": "__all", "view": "megatron_policy_wall"}
        )
    assert detail.status_code == aggregate.status_code == 200
    retained = {
        "0": {(WORKER, i) for i in range(25, 30)} | {("restart", i) for i in range(1, 4)},
        "1": {(WORKER, i) for i in range(54, 60)} | {("restart", i) for i in range(1, 3)},
    }
    wanted = [
        {
            "section": "megatron_phases",
            "execution_uid": row.execution_uid,
            "step": int(json.loads(row.attributes_json)["step"]),
            "rank": json.loads(row.attributes_json)["rank"],
            "phase": "ppo_train",
            "outcome": "success",
            "seconds": row.value,
        }
        for row in sorted(observed, key=lambda row: (-row.timestamp_ms, json.loads(row.attributes_json)["rank"]))
        if (row.execution_uid, int(json.loads(row.attributes_json)["step"]))
        in retained[json.loads(row.attributes_json)["rank"]]
    ]
    assert detail.json() == wanted
    # All 25 rank-step observations, including the older history excluded from detail.
    assert aggregate.json() == [
        {"section": "megatron_policy_wall", "t": BASE_EPOCH_MS, "series": "ppo_train max · all ranks", "value": 60},
        {
            "section": "megatron_policy_wall",
            "t": BASE_EPOCH_MS,
            "series": "ppo_train mean · all ranks",
            "value": pytest.approx(811 / 25),
        },
        {"section": "megatron_policy_wall", "t": BASE_EPOCH_MS, "series": "ppo_train min · all ranks", "value": 1},
    ]


def test_empty_telemetry_is_unknown_and_startup_only_runs_are_discoverable(rl_execution_store):
    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE name NOT IN ('lifecycle','terminal')")
    assert rl_execution_store.execute(resolve(run_variable_sql())).fetchall() == [("run",)]
    assert query(rl_execution_store, "Generated and consumed response tokens / s") == []
    assert query(rl_execution_store, "Rollouts completing during policy training") == []
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "status"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {
                "section": "status",
                "current_step": None,
                "steps_per_hour": None,
                "raw_reward": None,
                "generated_tokens_per_second": None,
                "since_last_step": None,
                "since_last_record": 239.999,
                "staleness_p95": None,
            }
        ]

    seed(
        rl_execution_store,
        [
            telemetry_row(
                "cuda_memory_observation",
                0,
                timestamp=WINDOW_START_MS + WINDOW_MS - 1000,
                attributes={"worker_role": "policy", "rank": "0", "gpu_uuid": "GPU-0", "phase": "forward"},
                body={"allocated_bytes": 1024, "peak_allocated_bytes": 2048, "device_total_bytes": 8192},
            )
        ],
    )
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "status"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {
                "section": "status",
                "current_step": None,
                "steps_per_hour": None,
                "raw_reward": None,
                "generated_tokens_per_second": None,
                "since_last_step": None,
                "since_last_record": 1,
                "staleness_p95": None,
            }
        ]


def test_core_gpu_hours_charge_both_roles_and_require_complete_counts(rl_execution_store):
    title = "Cumulative core GPU-hours in selected window"
    assert query(rl_execution_store, title)[0]["value"] == pytest.approx(10 * 16 / 3600)
    rl_execution_store.execute(
        f"DELETE FROM {TABLE} WHERE json_get(attributes_json,'metric')='async/performance/configured_inference_gpus'"
    )
    assert query(rl_execution_store, title) == []


def test_phase_service_rates_reject_clock_adjusted_windows(rl_execution_store):
    rl_execution_store.execute(
        f"UPDATE {TABLE} "
        "SET body_json=json_merge_patch(body_json,'{\"duration_seconds\":99}') WHERE name='async_phase_window'"
    )
    assert query(rl_execution_store, "Inference sampled throughput by learner phase") == []
