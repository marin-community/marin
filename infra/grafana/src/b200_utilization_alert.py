# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""B200 fleet activity alert from Kubernetes capacity and Finelog GPU telemetry."""

from datetime import datetime, timedelta

import pyarrow as pa
from k8s_source import Gb200Capacity

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
SELECT "cluster", node, COUNT(*) AS observed_gpus,
       SUM(CASE WHEN peak_utilization_percent > {GPU_ACTIVE_UTILIZATION_PERCENT} THEN 1 ELSE 0 END) AS active_gpus
FROM (
    SELECT "cluster", node_name AS node, json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           MAX(value) AS peak_utilization_percent
    FROM "telemetry_v1.node_agent"
    WHERE service = 'iris-node-agent' AND name = 'gpu_utilization_percent'
      AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
      AND node_name <> '' AND json_get(attributes_json, 'gpu_uuid') IS NOT NULL
    GROUP BY 1, 2, 3
) AS per_gpu
GROUP BY 1, 2
LIMIT {MAX_TELEMETRY_NODES + 1}
""".strip()


def b200_utilization_alert_rows(capacity: Gb200Capacity, telemetry: pa.Table) -> list[dict]:
    """Emit one row per cluster: `value` is 1 only for a measured idle cluster.

    `state` is a label, so Grafana shows why a cluster is not firing.
    """
    samples: dict[tuple[str, str], dict] = {}
    for row in telemetry.to_pylist():
        samples[(row["cluster"], row["node"])] = row

    results = []
    for cluster, gpus_by_node in capacity.ready_gpus.items():
        ready_gpus = sum(gpus_by_node.values())
        observed = 0
        active = 0
        for node, node_gpus in gpus_by_node.items():
            sample = samples.get((cluster, node))
            if sample is None:
                continue
            node_observed = min(int(sample["observed_gpus"]), node_gpus)
            observed += node_observed
            active += min(int(sample["active_gpus"]), node_observed)

        if cluster in capacity.unavailable:
            state = "capacity_unavailable"
        elif ready_gpus == 0:
            state = "no_ready_capacity"
        elif observed / ready_gpus < MIN_TELEMETRY_COVERAGE:
            state = "telemetry_incomplete"
        elif active / ready_gpus < MIN_ACTIVE_GPU_FRACTION:
            state = "idle"
        else:
            state = "active"
        results.append(
            {
                "cluster": cluster,
                "state": state,
                "value": int(state == "idle"),
                "active_percent": round(100 * active / ready_gpus, 1) if ready_gpus else 0.0,
                "ready_gpus": ready_gpus,
            }
        )
    return results
