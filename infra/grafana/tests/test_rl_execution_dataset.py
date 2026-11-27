# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import finelog_sql_for_duckdb
from rl_execution_fixtures import (
    BASE_EPOCH_MS,
    CLUSTER,
    DASHBOARD,
    DRIVER,
    ENDPOINT,
    PANELS,
    RUN_ID,
    TABLE,
    WINDOW_START_MS,
    WORKER,
    dataset,
    seed,
    telemetry_row,
)
from rl_execution_observability import (
    RL_EXECUTION_MAX_RESULT_ROWS,
)
from rl_execution_requests import (
    EXPECTED_PANELS,
    REQUEST,
    _bridge,
    _long_run_nodes,
    _long_run_store,
    _target_params,
)
from rl_observability import rl_gpu_dataset, rl_overview_dataset
from starlette.testclient import TestClient


def test_every_panel_view_returns_declared_fields_for_selected_execution(rl_execution_store):
    app, queries = _bridge(rl_execution_store)
    with TestClient(app) as client:
        assert client.get("/finelog/marin/v1/async-rl/overview").status_code == 404
        for run in (RUN_ID, "sync-run"):
            initial_queries = len(queries)
            for panel in PANELS.values():
                for target in panel["targets"]:
                    params = _target_params(target, run=run)
                    response = client.get(f"/finelog/marin{target['url']}", params=params)
                    assert response.status_code == 200, response.text
                    fixture = "status" if params["view"] == "status" else str(24 if panel["id"] == 158 else panel["id"])
                    expected = EXPECTED_PANELS[fixture]
                    columns = [column["selector"] for column in target["columns"]]
                    if fixture == "status":
                        assert set(columns) <= set(expected["columns"])
                    else:
                        assert columns == expected["columns"]
                    columns = expected["columns"]
                    returned = response.json()
                    assert all(set(row) == {"section", *columns} for row in returned)
                    assert all(row["section"] == params["view"] for row in returned)
                    actual = sorted([tuple(row[key] for key in columns) for row in returned], key=repr)
                    wanted_rows = expected.get("sync_rows", expected["rows"]) if run == "sync-run" else expected["rows"]
                    wanted = sorted([tuple(row) for row in wanted_rows], key=repr)
                    assert actual == [
                        tuple(pytest.approx(value, rel=1e-12) if isinstance(value, float) else value for value in row)
                        for row in wanted
                    ], (run, panel["id"], actual)
            assert len(queries) - initial_queries == len(dataset().sources), run

        panel = next(panel for panel in PANELS.values() if panel["id"] == 3)
        target = panel["targets"][0]
        response = client.get(
            f"/finelog/marin{target['url']}",
            params={**_target_params(target), "bucket_ms": 60000},
        )
        assert response.status_code == 200, response.text
        assert sorted(response.json(), key=lambda row: row["series"]) == [
            {"section": "token_rates", "t": BASE_EPOCH_MS, "series": "consumed_loss_token · driver", "value": 1.5},
            {
                "section": "token_rates",
                "t": BASE_EPOCH_MS,
                "series": "consumed_response_token · driver",
                "value": pytest.approx(100 / 60, rel=1e-12),
            },
            {"section": "token_rates", "t": BASE_EPOCH_MS, "series": "generated_token · driver", "value": 2.5},
        ]
        floored = client.get(
            f"/finelog/marin{target['url']}",
            params={**_target_params(target), "bucket_ms": 1000},
        )
        assert floored.status_code == 200, floored.text
        assert sorted(floored.json(), key=lambda row: row["series"]) == [
            {"section": "token_rates", "t": BASE_EPOCH_MS, "series": "consumed_loss_token · driver", "value": 3},
            {
                "section": "token_rates",
                "t": BASE_EPOCH_MS,
                "series": "consumed_response_token · driver",
                "value": pytest.approx(100 / 30, rel=1e-12),
            },
            {"section": "token_rates", "t": BASE_EPOCH_MS, "series": "generated_token · driver", "value": 5},
        ]

    seed(
        rl_execution_store,
        [
            telemetry_row(
                "policy_step",
                value,
                execution=execution,
                timestamp=BASE_EPOCH_MS + offset,
                resource={"training_type": "async"},
            )
            for execution, value, offset in [(DRIVER, 100, 1000), (WORKER, 2, 60000), (WORKER, 3, 120000)]
        ]
        + [
            telemetry_row(
                "training_metric_value",
                -0.25,
                timestamp=BASE_EPOCH_MS + 125000,
                attributes={"metric": "reward/avg_raw_reward", "payload_kind": "train", "role": "trainer"},
            ),
            telemetry_row(
                "work_completed", 240, timestamp=BASE_EPOCH_MS + 125000, attributes={"work_kind": "generated_token"}
            ),
        ],
    )
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "status"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {
                "section": "status",
                "current_step": 3,
                "steps_per_hour": 60,
                "raw_reward": -0.25,
                "generated_tokens_per_second": 8,
                "since_last_step": 120,
                "since_last_record": 115,
                "staleness_p95": pytest.approx(6.2),
            }
        ]

    rl_execution_store.execute(f"DELETE FROM {TABLE} WHERE run_id = 'sync-run' AND name = 'rollout_staleness_steps'")
    app, _ = _bridge(rl_execution_store)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "run": "sync-run", "view": "status"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {
                "section": "status",
                "current_step": 1,
                "steps_per_hour": None,
                "raw_reward": 0.5,
                "generated_tokens_per_second": 5,
                "since_last_step": 239.998,
                "since_last_record": 228,
                "staleness_p95": None,
            }
        ]


def test_all_executions_match_picker_values_and_keep_job_scope(telemetry_table):
    all_value = next(
        variable["allValue"] for variable in DASHBOARD["templating"]["list"] if variable["name"] == "execution"
    )
    rows = [telemetry_row("policy_step", i + 1, execution=f"execution-{i}") for i in range(3)]
    rows += [
        telemetry_row("policy_step", 1000, execution=None),
        telemetry_row("policy_step", 1000, execution="execution-0", job="other-job"),
    ]
    seed(telemetry_table, rows)
    app, _ = _bridge(telemetry_table)
    selected = {**REQUEST, "view": "policy_step"}
    expected = [
        {"section": "policy_step", "t": BASE_EPOCH_MS, "series": f"policy_step · execution-{i}", "value": i + 1}
        for i in range(3)
    ]
    with TestClient(app) as client:
        for executions in (all_value, "execution-0,execution-1,execution-2"):
            response = client.get(f"/finelog/marin{ENDPOINT}", params={**selected, "executions": executions})
            assert response.status_code == 200, response.text
            assert sorted(response.json(), key=lambda row: row["series"]) == expected
        seed(telemetry_table, [telemetry_row("policy_step", i + 1, execution=f"execution-{i}") for i in range(3, 35)])
        # A new app has a cold cache after the external source changes.
    app, _ = _bridge(telemetry_table)
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params={**selected, "executions": all_value})
        assert response.status_code == 200, response.text
        expected += [
            {"section": "policy_step", "t": BASE_EPOCH_MS, "series": f"policy_step · execution-{i}", "value": i + 1}
            for i in range(3, 35)
        ]
        assert sorted(response.json(), key=lambda row: row["series"]) == sorted(expected, key=lambda row: row["series"])
        explicit = client.get(
            f"/finelog/marin{ENDPOINT}",
            params={
                **selected,
                "executions": ",".join(f"execution-{i}" for i in range(35)),
            },
        )
        assert explicit.status_code == 400
        assert explicit.json() == {"error": "executions must contain between 1 and 32 values"}


def test_a_request_past_the_budget_asks_the_operator_to_narrow_it(rl_execution_store):
    app, _ = _bridge(rl_execution_store, max_rows=3)

    with TestClient(app) as client:
        capped = client.get(f"/finelog/marin{ENDPOINT}", params={**REQUEST, "view": "policy_step"})
        too_wide = client.get(
            f"/finelog/marin{ENDPOINT}",
            params={**REQUEST, "from": WINDOW_START_MS - 8 * 24 * 3600 * 1000, "view": "policy_step"},
        )

    assert capped.status_code == 400
    assert capped.json()["error"].endswith("narrow the RL execution filters or time range")
    assert too_wide.status_code == 400
    assert too_wide.json() == {"error": "RL execution range must not exceed 7 days"}
    window_ms = _long_run_store(rl_execution_store)
    selected = dataset(window_ms=window_ms, bucket_ms=30000)
    app, _ = _bridge(rl_execution_store, max_rows=RL_EXECUTION_MAX_RESULT_ROWS)
    long_params = {**REQUEST, "to": WINDOW_START_MS + window_ms}
    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{ENDPOINT}", params=long_params)
        assert response.status_code == 200, response.text
        assert 0 < len(response.json()) <= RL_EXECUTION_MAX_RESULT_ROWS
        assert 0 < len({row["t"] for row in response.json() if row["section"] == "policy_step"}) <= 360
        for source in selected.sources:
            rows = rl_execution_store.execute(finelog_sql_for_duckdb(source.sql)).fetch_arrow_table()
            assert 0 < rows.num_rows <= source.max_rows, (source.name, rows.num_rows, source.max_rows)
        for panel in PANELS.values():
            for target in panel["targets"]:
                params = {**_target_params(target), "to": str(WINDOW_START_MS + window_ms)}
                response = client.get(f"/finelog/marin{target['url']}", params=params)
                assert response.status_code == 200, (panel["id"], response.text)

    _long_run_nodes(rl_execution_store, 1)
    overview = rl_overview_dataset((CLUSTER,), RUN_ID, WINDOW_START_MS, WINDOW_START_MS + window_ms, 60000)
    gpu = rl_gpu_dataset((CLUSTER,), RUN_ID, WINDOW_START_MS, WINDOW_START_MS + 7 * 60 * 60 * 1000, 60000)
    sources = (*overview.sources, *gpu.sources)
    narrow_counts = {
        source.name: rl_execution_store.execute(finelog_sql_for_duckdb(source.sql)).fetch_arrow_table().num_rows
        for source in sources
    }
    _long_run_nodes(rl_execution_store, 8)
    for source in sources:
        rows = rl_execution_store.execute(finelog_sql_for_duckdb(source.sql)).fetch_arrow_table()
        assert 0 < rows.num_rows <= source.max_rows, (source.name, rows.num_rows, source.max_rows)
        assert rows.num_rows == narrow_counts[source.name] + (7 * 8 * 2 if source.name == "gpu" else 0)
    app, _ = _bridge(rl_execution_store, max_rows=RL_EXECUTION_MAX_RESULT_ROWS)
    with TestClient(app) as client:
        response = client.get("/finelog/marin/v1/rl/overview", params={**long_params, "bucket_ms": 60000})
        assert response.status_code == 200, response.text
        assert 0 < len(response.json()) <= overview.max_result_rows
        for view in overview.views:
            response = client.get(
                "/finelog/marin/v1/rl/overview", params={**long_params, "bucket_ms": 60000, "view": view}
            )
            assert response.status_code == 200, (view, response.text)
        for view in gpu.views:
            response = client.get("/finelog/marin/v1/rl/gpu", params={**long_params, "bucket_ms": 60000, "view": view})
            assert response.status_code == 200, (view, response.text)
            assert len(response.json()) == 1
            assert response.json()[0]["status"] == "range_limit"
