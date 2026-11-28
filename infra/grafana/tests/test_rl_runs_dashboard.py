# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import timedelta
from types import SimpleNamespace

import duckdb
import pytest
from config import ClusterTarget
from conftest import (
    bridge_config,
    dashboard_panels,
    finelog_sql_for_duckdb,
)
from rl_observability import rl_gpu_dataset, rl_overview_dataset
from rl_producers import producers_query
from rl_run_fixtures import (
    _COLUMNS,
    _NOW_MS,
    _SCHEMA,
    _WINDOW_START_MS,
    CLUSTER,
    NODES,
    RUN_ID,
    WINDOW_START,
    _census,
    _dashboard,
    _overview_rows,
    _panel_sql,
    _request_rows,
    _row,
    _run_rows,
)
from server import create_app
from starlette.testclient import TestClient


@pytest.mark.parametrize("build", [rl_overview_dataset, rl_gpu_dataset])
def test_selected_rl_views_preserve_combined_results_with_uneven_device_samples(rl_run_store, build):
    extra_samples = [
        _row(
            service="iris-node-agent",
            name="gpu_utilization_percent",
            value=0,
            moment=WINDOW_START + timedelta(seconds=sequence),
            seq=sequence,
            node_name=NODES[0],
            attributes={"gpu_uuid": f"GPU-{NODES[0]}-0"},
        )
        for sequence in (1, 2, 3)
    ]
    rl_run_store.executemany(
        f'INSERT INTO "telemetry_v1.node_agent" VALUES ({", ".join("?" for _ in _COLUMNS)})', extra_samples
    )
    combined = build((CLUSTER,), RUN_ID, _WINDOW_START_MS, _NOW_MS, 300_000)
    for view in combined.views:
        selected = build((CLUSTER,), RUN_ID, _WINDOW_START_MS, _NOW_MS, 300_000, view)
        results = []
        for dataset in (combined, selected):
            sources = finelog_sql_for_duckdb(
                ",\n".join(f"{source.name} AS ({source.sql})" for source in dataset.sources)
            )
            rows = rl_run_store.execute(f"WITH {sources}\n{dataset.views[view]}").fetchall()
            results.append(sorted(rows, key=lambda row: tuple(str(value) for value in row[:2])))
        assert len(results[0]) == len(results[1]), view
        for expected, actual in zip(results[0], results[1], strict=True):
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), view
        if view == "gpu_utilization":
            assert results[1][0] == (_WINDOW_START_MS, RUN_ID, 57.6875)


@pytest.mark.parametrize("view", ["critical_path", "straggler"])
def test_critical_path_sources_exclude_other_clocks_and_phases(rl_run_store, view):
    irrelevant = [
        _row(
            service="marinskyrl",
            name="phase_duration_seconds",
            value=1000,
            moment=WINDOW_START,
            seq=100,
            run_id=RUN_ID,
            attributes={"phase": phase, "clock_domain": clock, "outcome": "success"},
        )
        for phase, clock in (("train_step", "inclusive_wall"), ("rollout_call", "critical_path"))
    ]
    rl_run_store.executemany(
        f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({", ".join("?" for _ in _COLUMNS)})', irrelevant
    )
    selected = rl_overview_dataset((CLUSTER,), RUN_ID, _WINDOW_START_MS, _NOW_MS, 300_000, view)
    source_rows = rl_run_store.execute(finelog_sql_for_duckdb(selected.sources[0].sql)).fetch_arrow_table().to_pylist()
    assert {(row["clock_domain"], row["phase"]) for row in source_rows} == {
        ("critical_path", "rollout_or_inference_wait"),
        ("critical_path", "train_step"),
    }
    combined = rl_overview_dataset((CLUSTER,), RUN_ID, _WINDOW_START_MS, _NOW_MS, 300_000)
    results = []
    for dataset in (combined, selected):
        sources = finelog_sql_for_duckdb(",\n".join(f"{source.name} AS ({source.sql})" for source in dataset.sources))
        rows = rl_run_store.execute(f"WITH {sources}\n{dataset.views[view]}").fetchall()
        results.append(sorted(rows, key=lambda row: tuple(str(value) for value in row[:2])))
    assert results[0] == results[1]


def test_the_trainer_panels_render_for_that_run(rl_run_store) -> None:
    # A failed step ran 0.5s before raising; a successful one takes 6s. Blending them would
    # report 3.25s, which describes neither.
    phases = rl_run_store.execute(_panel_sql("Rollout wait vs train step (critical path)")).fetchall()
    by_series = {row[1]: row[2] for row in phases}
    assert by_series["rollout_or_inference_wait · success"] == pytest.approx(44.0)
    assert by_series["train_step · success"] == pytest.approx(6.0)
    assert by_series["train_step · failure"] == pytest.approx(0.5)

    # Occupancy is a ratio, so two nodes reporting 3 GB used of 4 GB still reads 0.75 rather
    # than doubling. 6e9 used over 8e9 total.
    occupancy = rl_run_store.execute(_panel_sql("Ray object store occupancy")).fetchall()
    assert [row[1] for row in occupancy] == [pytest.approx(0.75)] * 6


def test_percentile_panels_compute_over_all_executions_in_each_bucket(rl_run_store) -> None:
    rl_run_store.execute(
        "DELETE FROM \"telemetry_v1.marinskyrl\" WHERE name='phase_duration_seconds' "
        "AND json_get(attributes_json, 'phase')='rollout_or_inference_wait'"
    )
    rows = []
    for seq, (minutes, execution, value) in enumerate(
        [
            (0, "attempt-a", 0),
            (5, "attempt-a", 100),
            (10, "attempt-b", 10),
            (15, "attempt-b", 10),
            (20, "attempt-b", 10),
            (25, "attempt-b", 10),
            (29, "attempt-b", 10),
            (35, "attempt-a", 20),
            (40, "attempt-b", 40),
        ]
    ):
        rows.append(
            _row(
                service="marinskyrl",
                name="phase_duration_seconds",
                value=value,
                moment=WINDOW_START + timedelta(minutes=minutes),
                seq=seq,
                run_id=RUN_ID,
                execution_uid=execution,
                attributes={"phase": "rollout_or_inference_wait", "clock_domain": "critical_path", "outcome": "success"},
            )
        )
    rl_run_store.executemany(f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({", ".join("?" for _ in _COLUMNS)})', rows)
    assert _overview_rows(rl_run_store, "straggler") == [
        {"section": "straggler", "t": _WINDOW_START_MS, "series": "p99 / p50", "value": pytest.approx(94.6 / 10)},
        {
            "section": "straggler",
            "t": _WINDOW_START_MS + 1800000,
            "series": "p99 / p50",
            "value": pytest.approx(39.8 / 30),
        },
    ]


def _assert_gpu_view_rows(client, params, panels, expected):
    for end in (_NOW_MS, _WINDOW_START_MS + 7 * 60 * 60 * 1000):
        for panel in panels:
            (target,) = panel["targets"]
            target_params = {
                param["key"]: str(params[param["key"]]) if param["key"] in params else param["value"]
                for param in target["url_options"]["params"]
            }
            target_params["to"] = str(end)
            response = client.get(f"/finelog/marin{target['url']}", params=target_params)
            assert response.status_code == 200, response.text
            wanted = expected[target_params["view"]]
            assert sorted(response.json(), key=lambda row: (row.get("t", 0), row.get("series", ""))) == sorted(
                wanted, key=lambda row: (row.get("t", 0), row.get("series", ""))
            )


def _assert_gpu_query_isolation(rl_run_store, params, panels, expected, source):
    overview = _request_rows(rl_run_store, "/v1/rl/overview", params)
    assert overview
    rl_run_store.execute('UPDATE "telemetry_v1.marinskyrl" SET node_name = NULL')
    for view in expected:
        assert _overview_rows(rl_run_store, view, "/v1/rl/gpu") == []
    rl_run_store.execute('DROP TABLE "telemetry_v1.node_agent"')
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    message = "Zoom to 7h or less for GPU detail."
    capped = {
        view: {
            "section": view,
            "status": "range_limit",
            "message": message,
            **(
                {"t": _WINDOW_START_MS, "mean_used_bytes": None, "peak_used_bytes": None}
                if view == "gpu_memory"
                else (
                    {"node": message, "gpu": None, "peak_power_watts": None, "pcie_replay_increase": None}
                    if view == "pcie_faults"
                    else {"t": _WINDOW_START_MS, "series": message, "value": None}
                )
            ),
        }
        for view in expected
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.get("/finelog/marin/v1/rl/gpu", params=params)
        assert failed.status_code == 500
        healthy = client.get("/finelog/marin/v1/rl/overview", params=params)
        assert healthy.status_code == 200, healthy.text
        assert sorted(healthy.json(), key=lambda row: json.dumps(row, sort_keys=True)) == sorted(
            overview, key=lambda row: json.dumps(row, sort_keys=True)
        )
        for end in (_WINDOW_START_MS + 7 * 60 * 60 * 1000 + 1, _WINDOW_START_MS + 7 * 24 * 60 * 60 * 1000):
            for panel in panels:
                (target,) = panel["targets"]
                view = next(param["value"] for param in target["url_options"]["params"] if param["key"] == "view")
                response = client.get(f"/finelog/marin{target['url']}", params={**params, "to": end, "view": view})
                assert response.status_code == 200, response.text
                assert response.json() == [capped[view]]


@pytest.mark.parametrize("contended", [False, True])
def test_the_node_agent_joins_through_node_name_without_a_run_id(rl_run_store, contended) -> None:
    distractors = [
        _row(
            service="iris-node-agent",
            name=name,
            value=value,
            moment=moment,
            seq=0,
            node_name=node,
            attributes={"gpu_uuid": gpu},
        )
        for node, gpu, moment, name, value in [
            (NODES[0], f"GPU-{NODES[0]}-0", WINDOW_START - timedelta(seconds=1), "gpu_pcie_replay_errors", 10000),
            ("other-node", "other-gpu", WINDOW_START, "gpu_sm_active_ratio", 1000),
            ("other-node", "other-gpu", WINDOW_START, "gpu_pcie_replay_errors", 1000),
        ]
    ]
    rl_run_store.executemany(
        f"""INSERT INTO "telemetry_v1.node_agent" VALUES ({', '.join('?' for _ in _COLUMNS)})""", distractors
    )
    if contended:
        competing = [
            _row(
                service="marinskyrl",
                name="process_cpu_percent",
                value=1,
                moment=WINDOW_START + timedelta(minutes=5 * bucket),
                seq=seq,
                run_id="another-run",
                node_name=NODES[0],
            )
            for bucket in (1, 2)
            for seq in range(100)
        ]
        rl_run_store.executemany(
            f"""INSERT INTO "telemetry_v1.marinskyrl" VALUES ({', '.join('?' for _ in _COLUMNS)})""", competing
        )
    times = [_WINDOW_START_MS + bucket * 300000 for bucket in range(6)]
    expected = {
        "gpu_utilization": [{"t": t, "series": RUN_ID, "value": 71 + bucket} for bucket, t in enumerate(times)],
        "sm_activity": [
            {"t": t, "series": series, "value": pytest.approx(value)}
            for t in times
            for series, value in (("SM active", 60), ("tensor pipe active", 30))
        ],
        "gpu_memory": [{"t": t, "mean_used_bytes": 50000000000.0, "peak_used_bytes": 60000000000.0} for t in times],
        "fabric_receive": [
            {"t": t, "series": series, "value": value}
            for bucket, t in enumerate(times)
            for series, value in (
                ("NVLink receive", 200 if contended and bucket in (1, 2) else 400),
                ("PCIe receive", 20 if contended and bucket in (1, 2) else 40),
            )
        ],
        "pcie_faults": [
            {
                "node": NODES[0],
                "gpu": f"GPU-{NODES[0]}-0",
                "peak_power_watts": 620,
                "pcie_replay_increase": 1 if contended else 5,
            }
        ],
    }
    params = {"clusters": CLUSTER, "run": RUN_ID, "from": _WINDOW_START_MS, "to": _NOW_MS, "bucket_ms": 300000}
    panels = [
        panel
        for panel in dashboard_panels(_dashboard()["panels"])
        if any(target["url"] == "/v1/rl/gpu" for target in panel.get("targets", []))
    ]
    for view, wanted in expected.items():
        for row in wanted:
            row["section"] = view

    def query(sql, *, max_rows):
        return rl_run_store.execute(finelog_sql_for_duckdb(sql)).fetch_arrow_table()

    source = SimpleNamespace(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query)
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    with TestClient(app) as client:
        _assert_gpu_view_rows(client, params, panels, expected)
    _assert_gpu_query_isolation(rl_run_store, params, panels, expected, source)


def test_the_census_still_answers_on_a_deployment_that_has_no_harbor_namespace(rl_run_store) -> None:
    """A deployment without every RL namespace is the normal case, so the census leaves it out."""
    rows = _census(rl_run_store, frozenset({"telemetry_v1.marinskyrl", "telemetry_v1.vllm"}))

    assert {row["producer"] for row in rows} == {"marinskyrl", "vllm"}


def test_the_census_is_empty_rather_than_broken_on_a_deployment_holding_none_of_them() -> None:
    assert _census(duckdb.connect(), frozenset()) == []


def test_the_census_quotes_a_run_id_carrying_an_apostrophe() -> None:
    database = duckdb.connect()
    database.execute(f'CREATE TABLE "telemetry_v1.marinskyrl"{_SCHEMA}')
    database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
    run_index, service_index = _COLUMNS.index("run_id"), _COLUMNS.index("service")
    row = next(row for row in _run_rows() if row[service_index] == "marinskyrl")
    placeholders = ", ".join("?" for _ in _COLUMNS)
    database.execute(
        f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({placeholders})',
        (*row[:run_index], "run's-id", *row[run_index + 1 :]),
    )

    sql = producers_query("telemetry_v1.marinskyrl", "run's-id", (CLUSTER,), _WINDOW_START_MS, _NOW_MS)

    assert len(database.execute(sql).fetchall()) == 1
