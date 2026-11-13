# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""B200 idle alerts require both Ready capacity and sufficiently complete GPU telemetry."""

from datetime import UTC, datetime, timedelta

import duckdb
import pyarrow as pa
from b200_utilization_alert import b200_utilization_alert_rows, b200_utilization_query
from conftest import install_finelog_dialect_macros, k8s_api, make_k8s_source, node
from k8s_source import Gb200Capacity, K8sFleet

NOW = datetime(2026, 11, 9, 12, tzinfo=UTC)
GB200_CLUSTER = "cw-us-east-08a"


def _gb200(name: str, *, ready: bool = True) -> dict:
    return node(name, instance_type="gb200-4x", gpu_capacity=4, ready=ready)


def _capacity(*gb200_cluster_nodes: dict) -> Gb200Capacity:
    """Capacity of a two-cluster fleet: one cluster with no nodes and one with the given nodes."""
    fleet = K8sFleet(
        [
            make_k8s_source(k8s_api({"/api/v1/nodes": []}), name="cw-us-east-02a"),
            make_k8s_source(k8s_api({"/api/v1/nodes": list(gb200_cluster_nodes)}), name=GB200_CLUSTER),
        ]
    )
    return fleet.gb200_capacity(fleet.nodes())


def _telemetry(*rows: tuple[str, int, int]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"cluster": GB200_CLUSTER, "node": name, "observed_gpus": observed, "active_gpus": active}
            for name, observed, active in rows
        ]
    )


def test_b200_alert_fires_for_idle_ready_capacity_and_ignores_other_hardware():
    capacity = _capacity(
        _gb200("gb200-a"),
        _gb200("gb200-b"),
        _gb200("unready", ready=False),
        node("h100", instance_type="gd-8xh100ib-i128", gpu_capacity=8),
    )
    rows = b200_utilization_alert_rows(
        capacity, _telemetry(("gb200-a", 4, 0), ("gb200-b", 4, 0), ("unready", 4, 4), ("h100", 8, 8))
    )

    assert rows == [
        {"cluster": "cw-us-east-02a", "state": "no_ready_capacity", "value": 0, "active_percent": 0.0, "ready_gpus": 0},
        {"cluster": GB200_CLUSTER, "state": "idle", "value": 1, "active_percent": 0.0, "ready_gpus": 8},
    ]


def test_b200_alert_stays_normal_with_active_gpus_or_incomplete_telemetry():
    capacity = _capacity(_gb200("gb200-a"), _gb200("gb200-b"))
    active = b200_utilization_alert_rows(capacity, _telemetry(("gb200-a", 4, 1), ("gb200-b", 4, 0)))
    missing = b200_utilization_alert_rows(capacity, _telemetry(("gb200-a", 4, 0)))

    assert active[1] == {
        "cluster": GB200_CLUSTER,
        "state": "active",
        "value": 0,
        "active_percent": 12.5,
        "ready_gpus": 8,
    }
    assert missing[1]["state"] == "telemetry_incomplete" and missing[1]["value"] == 0


def test_b200_alert_distinguishes_inventory_failure_from_zero_capacity():
    fleet = K8sFleet([make_k8s_source(k8s_api({}), name=GB200_CLUSTER)])
    rows = b200_utilization_alert_rows(fleet.gb200_capacity(fleet.nodes()), pa.table({}))

    assert rows == [
        {"cluster": GB200_CLUSTER, "state": "capacity_unavailable", "value": 0, "active_percent": 0.0, "ready_gpus": 0}
    ]


def test_b200_query_counts_a_gpu_as_active_on_its_peak_recent_sample():
    database = duckdb.connect()
    install_finelog_dialect_macros(database)
    database.execute(
        'CREATE TABLE "telemetry_v1.node_agent"(cluster VARCHAR, node_name VARCHAR, service VARCHAR, '
        "name VARCHAR, attributes_json VARCHAR, value DOUBLE, timestamp_ms BIGINT)"
    )
    recent = int((NOW - timedelta(minutes=1)).timestamp() * 1000)
    old = int((NOW - timedelta(minutes=6)).timestamp() * 1000)

    def sample(gpu: str, value: float, timestamp_ms: int, name: str = "gpu_utilization_percent") -> tuple:
        return (GB200_CLUSTER, "gb200-a", "iris-node-agent", name, f'{{"gpu_uuid":"{gpu}"}}', value, timestamp_ms)

    database.executemany(
        'INSERT INTO "telemetry_v1.node_agent" VALUES (?, ?, ?, ?, ?, ?, ?)',
        [
            # GPU a was busy only before the lookback; GPU b had one busy sample inside it.
            sample("a", 99, old),
            sample("a", 3, recent - 1000),
            sample("a", 2, recent),
            sample("b", 75, recent - 1000),
            sample("b", 1, recent),
            sample("c", 400, recent, name="gpu_power_watts"),
        ],
    )

    assert database.execute(b200_utilization_query(NOW)).fetchall() == [(GB200_CLUSTER, "gb200-a", 2, 1)]
