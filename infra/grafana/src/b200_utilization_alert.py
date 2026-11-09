# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""B200 fleet activity alert from Kubernetes capacity and Finelog GPU telemetry."""

from datetime import datetime, timedelta

import pyarrow as pa

GPU_ACTIVE_UTILIZATION_PERCENT = 10
MIN_ACTIVE_GPU_FRACTION = 0.1
MIN_TELEMETRY_COVERAGE = 0.8
TELEMETRY_LOOKBACK = timedelta(minutes=5)
MAX_TELEMETRY_NODES = 5000


def b200_utilization_query(now: datetime) -> str:
    """Return recent per-node GPU counts and activity from the hub's node-agent telemetry."""
    start_ms = int((now - TELEMETRY_LOOKBACK).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    return f"""
WITH latest AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS cluster,
           node_name AS node,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           value AS utilization_percent,
           ROW_NUMBER() OVER (
               PARTITION BY cluster, node_name, json_get(attributes_json, 'gpu_uuid')
               ORDER BY timestamp_ms DESC
           ) AS rn
    FROM "telemetry_v1.node_agent"
    WHERE service = 'iris-node-agent' AND name = 'gpu_utilization_percent'
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
      AND node_name <> '' AND json_get(attributes_json, 'gpu_uuid') IS NOT NULL
)
SELECT cluster, node, COUNT(*) AS observed_gpus,
       SUM(CASE WHEN utilization_percent > {GPU_ACTIVE_UTILIZATION_PERCENT} THEN 1 ELSE 0 END) AS active_gpus
FROM latest WHERE rn = 1 GROUP BY 1, 2
LIMIT {MAX_TELEMETRY_NODES + 1}
""".strip()


def b200_utilization_alert_rows(nodes: list[dict], telemetry: pa.Table, clusters: tuple[str, ...]) -> list[dict]:
    """Emit active GPU percentages, withholding idle decisions when telemetry is incomplete."""
    ready: dict[str, dict[str, int]] = {cluster: {} for cluster in clusters}
    for row in nodes:
        cluster = row.get("cluster")
        if cluster not in ready or "gb200" not in row.get("instance_type", "").lower():
            continue
        if row.get("ready") and not row.get("unschedulable") and row.get("gpu_capacity", 0) > 0:
            ready[cluster][row["node"]] = row["gpu_capacity"]

    samples: dict[tuple[str, str], dict] = {}
    for row in telemetry.to_pylist():
        samples[(row["cluster"], row["node"])] = row

    results = []
    for cluster, capacity_by_node in ready.items():
        capacity = sum(capacity_by_node.values())
        observed = 0
        active = 0
        for node, node_capacity in capacity_by_node.items():
            sample = samples.get((cluster, node))
            if sample is None:
                continue
            node_observed = min(int(sample["observed_gpus"]), node_capacity)
            observed += node_observed
            active += min(int(sample["active_gpus"]), node_observed)

        if capacity == 0:
            state = "no_ready_capacity"
        elif observed / capacity < MIN_TELEMETRY_COVERAGE:
            state = "telemetry_incomplete"
        elif active / capacity < MIN_ACTIVE_GPU_FRACTION:
            state = "idle"
        else:
            state = "active"
        results.append(
            {
                "cluster": cluster,
                "state": state,
                "ready_gpus": str(capacity),
                "value": round(100 * active / capacity, 1) if state in ("idle", "active") else 100.0,
            }
        )
    return results
