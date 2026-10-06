# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU history sampling, uncertainty, and shared bridge reads."""

from datetime import UTC, datetime
from types import SimpleNamespace

import duckdb
import pyarrow as pa
from config import ClusterTarget
from conftest import bridge_config, install_finelog_dialect_macros
from errors import UpstreamError
from gpu_allocation_history import DAY_MS, history_rows, live_rows, metadata_table
from server import create_app
from starlette.testclient import TestClient

CLUSTER = "cw-rno2a"
TARGET = ClusterTarget("marin", "project", "zone", "finelog", "controller")


def _time(ms):
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _attempt(
    task,
    gpus,
    started,
    finished=None,
    *,
    created=0,
    model="H100",
    requested="INTERACTIVE",
    applied="BATCH",
    attempt=0,
    current=0,
):
    row = {
        "rootJobId": "/u/root",
        "taskId": task,
        "gpuCount": gpus,
        "gpuVariant": model,
        "requestedPriority": "PRIORITY_BAND_" + requested,
        "currentAppliedPriority": "PRIORITY_BAND_" + applied,
        "attemptId": attempt,
        "currentAttemptId": current,
        "createdAtMs": str(created),
    }
    if started is not None:
        row["startedAtMs"] = str(started)
    if finished is not None:
        row["finishedAtMs"] = str(finished)
    return row


def _state(points):
    schema = pa.schema(
        [
            ("point", pa.timestamp("ms", tz="UTC")),
            ("ts", pa.timestamp("ms", tz="UTC")),
            ("cluster", pa.string()),
            ("root_job_id", pa.string()),
            ("assigned", pa.int64()),
            ("building", pa.int64()),
            ("running", pa.int64()),
        ]
    )
    rows = []
    for point, ts, running, building in points:
        for root in ("", "/u/root"):
            rows.append(
                {
                    "point": _time(point),
                    "ts": _time(ts),
                    "cluster": CLUSTER,
                    "root_job_id": root,
                    "assigned": 0,
                    "building": building,
                    "running": running,
                }
            )
    return pa.Table.from_pylist(rows, schema=schema)


def test_history_preserves_retry_boundaries_child_shapes_and_unknown_old_priority():
    metadata = metadata_table(
        [
            _attempt("/u/root/0", 0, 1, applied="INTERACTIVE"),
            _attempt("/u/root/two/0", 2, 20_000, 90_000, current=1),
            _attempt("/u/root/two/0", 2, 90_000, applied="INTERACTIVE", attempt=1, current=1, created=90_000),
            _attempt("/u/root/eight/0", 8, 80_000, 120_000, created=80_000, requested="BATCH"),
            _attempt("/u/root/setup/0", 4, 170_000, created=100_000),
        ],
        0,
        DAY_MS,
    )
    state = _state([(60_000, 59_000, 2, 0), (120_000, 119_000, 3, 1)])
    rows = history_rows(state, {(0, CLUSTER): metadata}, {}, 60_000, 180_000, 60_000, (CLUSTER,))
    h100 = [r for r in rows if r["model"] == "H100"]
    assert h100[0]["unknown_priority"] == 2
    assert h100[0]["batch"] == 0
    assert h100[1]["interactive"] == 2
    assert h100[1]["batch"] == 0  # The eight-GPU attempt ended exactly at this instant.
    assert h100[1]["unknown_priority"] == 0  # Setup is not unknown-priority allocation.
    assert h100[1]["setup_gpu_requests"] == 4
    assert all(r["idle"] is None and r["incomplete"] == 1 for r in rows)
    assert all(r["missing_task_metadata"] == 0 for r in rows)


def test_history_does_not_extend_a_cached_attempt_into_a_day_with_missing_metadata():
    rows = history_rows(
        _state([(DAY_MS, DAY_MS - 1000, 1, 0)]),
        {(0, CLUSTER): metadata_table([_attempt("/u/root/0", 8, 1)], 0, DAY_MS)},
        {(DAY_MS, CLUSTER): "unavailable"},
        DAY_MS,
        DAY_MS + 60_000,
        60_000,
        (CLUSTER,),
    )
    assert all(row["interactive"] is None and row["batch"] is None for row in rows)
    assert all(row["missing_clusters"] == CLUSTER for row in rows)
    assert all(row["idle"] is None for row in rows)


def test_history_does_not_assign_unresolved_gpu_models_to_a_priority_or_idle_band():
    rows = history_rows(
        _state([(60_000, 59_000, 0, 1)]),
        {(0, CLUSTER): metadata_table([_attempt("/u/root/0", 8, None, model="auto")], 0, DAY_MS)},
        {},
        60_000,
        120_000,
        60_000,
        (CLUSTER,),
    )
    assert all(row["unknown_model_gpu_requests"] == 8 for row in rows)
    assert all(row["unknown_priority"] == 0 and row["batch"] == 0 and row["idle"] is None for row in rows)


def test_history_does_not_project_counts_past_the_metadata_observation_range():
    rows = history_rows(
        _state([(120_000, 119_000, 1, 0)]),
        {(0, CLUSTER): metadata_table([_attempt("/u/root/0", 8, 1)], 0, 119_000)},
        {},
        120_000,
        180_000,
        60_000,
        (CLUSTER,),
    )
    assert all(row["batch"] is None and row["idle"] is None for row in rows)


def test_live_allocation_includes_bound_setup_and_pods_after_controller_finish():
    nodes = [
        {
            "cluster": CLUSTER,
            "node": "gpu-node",
            "gpu_model": "H100_NVLINK_80GB",
            "gpu_capacity": 16,
            "gpu_allocatable": 16,
        }
    ]
    pods = [
        {"cluster": CLUSTER, "node": node, "gpu_request_count": gpus, "phase": phase, "priority_class": priority}
        for node, gpus, phase, priority in [
            ("gpu-node", 4, "Pending", "iris-batch"),
            ("gpu-node", 8, "Running", "iris-interactive"),
            ("", 8, "Pending", "iris-interactive"),
            ("gpu-node", 8, "Succeeded", "iris-interactive"),
        ]
    ]
    row = next(r for r in live_rows(nodes, pods, (CLUSTER,), 60_000) if r["model"] == "H100")
    assert row["batch"] == 4 and row["interactive"] == 8
    assert row["allocated"] == 12 and row["idle"] == 4 and row["capacity"] == 16
    assert row["incomplete"] == 0


def test_live_source_failure_does_not_fill_missing_capacity_with_idle():
    error = {"cluster": CLUSTER, "error_class": "auth", "error": "denied"}
    assert all(row["idle"] is None and row["incomplete"] == 1 for row in live_rows([error], [error], (CLUSTER,), 60_000))


def _database_source(database):
    queries = []

    def query(sql, *, max_rows):
        queries.append(sql)
        table = database.execute(sql).fetch_arrow_table()
        assert table.num_rows <= max_rows
        return table

    return SimpleNamespace(target=TARGET, query=query), queries


class Registry:
    def __init__(self, rows):
        self.rows = rows
        self.requests = []

    def gpu_allocation_metadata(self, cluster, start, end, *, max_rows):
        self.requests.append((cluster, start, end, max_rows))
        return self.rows


def _client(source, registry):
    return TestClient(create_app(bridge_config(), {"marin": source}, {"marin": registry}, None, None, None))


def test_bridge_reads_whole_emissions_and_shares_inputs_between_both_models():
    with duckdb.connect() as database:
        install_finelog_dialect_macros(database)
        database.execute(
            """CREATE TABLE "iris.task_state" (ts TIMESTAMP, cluster VARCHAR, root_job_id VARCHAR,
               assigned BIGINT, building BIGINT, running BIGINT)"""
        )
        for ts, root in [(20_000, "/u/old"), (50_000, "/u/root")]:
            for job in ("", root):
                database.execute('INSERT INTO "iris.task_state" VALUES (?,?,?,0,0,1)', [_time(ts), CLUSTER, job])
        source, queries = _database_source(database)
        registry = Registry(
            [
                _attempt("/u/root/0", 4, 50_000),
                {**_attempt("/u/old/0", 8, 10_000, 40_000), "rootJobId": "/u/old"},
            ]
        )
        with _client(source, registry) as client:
            params = {"from": 60_000, "to": 120_000, "clusters": CLUSTER}
            first = client.get("/finelog/marin/v1/gpu/allocation", params={**params, "model": "H100"})
            second = client.get("/finelog/marin/v1/gpu/allocation", params={**params, "model": "GB200"})
            assert first.status_code == second.status_code == 200
            assert first.json()[0]["batch"] == 4
            assert first.json()[0]["missing_task_metadata"] == 0
            assert second.json()[0]["batch"] == 0
            assert len(queries) == len(registry.requests) == 1


def test_bridge_refresh_reuses_closed_days_when_the_week_window_moves(monkeypatch):
    start = 20_000 * DAY_MS + 60_000
    end = start + 7 * DAY_MS
    clock = SimpleNamespace(now=100.0, epoch=end)
    monkeypatch.setattr("cache.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("server.time.time_ns", lambda: clock.epoch * 1_000_000)
    with duckdb.connect() as database:
        install_finelog_dialect_macros(database)
        database.execute(
            """CREATE TABLE "iris.task_state" (ts TIMESTAMP, cluster VARCHAR, root_job_id VARCHAR,
               assigned BIGINT, building BIGINT, running BIGINT)"""
        )
        source, queries = _database_source(database)
        registry = Registry([_attempt("/u/root/0", 4, start - 60_000)])
        with _client(source, registry) as client:
            params = {"from": start, "to": end, "clusters": CLUSTER, "model": "H100"}
            response = client.get("/finelog/marin/v1/gpu/allocation", params=params)
            assert response.status_code == 200 and len(response.json()) == 672
            first_reads = len(queries)
            assert first_reads == len(registry.requests) == 8
            clock.now += 21
            clock.epoch += 60_000
            response = client.get(
                "/finelog/marin/v1/gpu/allocation", params={**params, "from": start + 60_000, "to": end + 60_000}
            )
            assert response.status_code == 200 and len(response.json()) == 672
            assert len(queries) == len(registry.requests) == first_reads + 1


def test_bridge_reports_regional_access_failure_without_manufacturing_idle():
    class UnavailableRegistry(Registry):
        def gpu_allocation_metadata(self, *args, **kwargs):
            raise UpstreamError("iris", "regional metadata RPC unavailable")

    with duckdb.connect() as database:
        install_finelog_dialect_macros(database)
        database.execute(
            """CREATE TABLE "iris.task_state" (ts TIMESTAMP, cluster VARCHAR, root_job_id VARCHAR,
               assigned BIGINT, building BIGINT, running BIGINT)"""
        )
        source, _ = _database_source(database)
        with _client(source, UnavailableRegistry([])) as client:
            response = client.get(
                "/finelog/marin/v1/gpu/allocation", params={"from": 60_000, "to": 120_000, "clusters": CLUSTER}
            )
            assert response.status_code == 200
            assert all(
                r["missing_clusters"] == CLUSTER and r["batch"] is None and r["idle"] is None for r in response.json()
            )
