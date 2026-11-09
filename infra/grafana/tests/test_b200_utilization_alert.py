# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""B200 idle alerts require both Ready capacity and sufficiently complete GPU telemetry."""

from datetime import UTC, datetime, timedelta

import duckdb
import pyarrow as pa
from b200_utilization_alert import b200_capacity_alert_rows, b200_utilization_alert_rows, b200_utilization_query
from conftest import install_finelog_dialect_macros

NOW = datetime(2026, 11, 9, 12, tzinfo=UTC)
CLUSTERS = ("cw-us-east-02a", "cw-us-east-08a")


def _node(name: str, *, cluster: str = "cw-us-east-08a", capacity: int = 4, ready: bool = True) -> dict:
    return {
        "cluster": cluster,
        "node": name,
        "instance_type": "gb200-4x",
        "gpu_capacity": capacity,
        "ready": ready,
        "unschedulable": False,
    }


def _telemetry(*rows: tuple[str, int, int]) -> pa.Table:
    return (
        pa.Table.from_pylist(
            [
                {
                    "cluster": "cw-us-east-08a",
                    "node": node,
                    "observed_gpus": observed,
                    "active_gpus": active,
                }
                for node, observed, active in rows
            ]
        )
        if rows
        else pa.table({})
    )


def test_b200_alert_fires_for_idle_ready_capacity_and_ignores_other_hardware():
    nodes = [_node("gb200-a"), _node("gb200-b"), _node("unready", ready=False)]
    nodes.append({**_node("h100", capacity=8), "instance_type": "gd-8xh100ib-i128"})
    rows = b200_utilization_alert_rows(
        nodes,
        _telemetry(("gb200-a", 4, 0), ("gb200-b", 4, 0), ("unready", 4, 4), ("h100", 8, 8)),
        CLUSTERS,
    )

    assert rows[1] == {
        "cluster": "cw-us-east-08a",
        "state": "idle",
        "value": 0.0,
    }
    assert rows[0]["state"] == "no_ready_capacity" and rows[0]["value"] == 100.0
    assert b200_capacity_alert_rows(nodes, CLUSTERS) == [
        {"cluster": "cw-us-east-02a", "value": 0},
        {"cluster": "cw-us-east-08a", "value": 8},
    ]


def test_b200_alert_stays_normal_with_active_gpus_or_incomplete_telemetry():
    nodes = [_node("gb200-a"), _node("gb200-b")]
    active = b200_utilization_alert_rows(nodes, _telemetry(("gb200-a", 4, 1), ("gb200-b", 4, 0)), CLUSTERS)
    missing = b200_utilization_alert_rows(nodes, _telemetry(("gb200-a", 4, 0)), CLUSTERS)

    assert active[1]["state"] == "active" and active[1]["value"] == 12.5
    assert missing[1]["state"] == "telemetry_incomplete" and missing[1]["value"] == 100.0


def test_b200_alert_distinguishes_inventory_failure_from_zero_capacity():
    rows = b200_utilization_alert_rows([{"cluster": "cw-us-east-08a", "error_class": "network"}], pa.table({}), CLUSTERS)

    assert rows[1] == {"cluster": "cw-us-east-08a", "state": "capacity_unavailable", "value": 100.0}


def test_b200_query_uses_latest_recent_sample_per_gpu():
    database = duckdb.connect()
    install_finelog_dialect_macros(database)
    database.execute(
        'CREATE TABLE "telemetry_v1.node_agent"(cluster VARCHAR, node_name VARCHAR, service VARCHAR, '
        "name VARCHAR, attributes_json VARCHAR, value DOUBLE, timestamp_ms BIGINT)"
    )
    recent = int((NOW - timedelta(minutes=1)).timestamp() * 1000)
    old = int((NOW - timedelta(minutes=6)).timestamp() * 1000)
    database.executemany(
        'INSERT INTO "telemetry_v1.node_agent" VALUES (?, ?, ?, ?, ?, ?, ?)',
        [
            ("cw-us-east-08a", "gb200-a", "iris-node-agent", "gpu_utilization_percent", '{"gpu_uuid":"a"}', 99, old),
            (
                "cw-us-east-08a",
                "gb200-a",
                "iris-node-agent",
                "gpu_utilization_percent",
                '{"gpu_uuid":"a"}',
                80,
                recent - 1000,
            ),
            ("cw-us-east-08a", "gb200-a", "iris-node-agent", "gpu_utilization_percent", '{"gpu_uuid":"a"}', 2, recent),
            ("cw-us-east-08a", "gb200-a", "iris-node-agent", "gpu_utilization_percent", '{"gpu_uuid":"b"}', 75, recent),
            ("cw-us-east-08a", "gb200-a", "iris-node-agent", "gpu_power_watts", '{"gpu_uuid":"c"}', 400, recent),
        ],
    )

    assert database.execute(b200_utilization_query(NOW)).fetchall() == [("cw-us-east-08a", "gb200-a", 2, 1)]
