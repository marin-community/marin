# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded source families shared by fleet accelerator panels."""

import math

from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_values
from vllm_observability import sql_values

ACCELERATOR_MAX_WINDOW_MS = 24 * 60 * 60 * 1000
ACCELERATOR_MAX_POINTS = 100
ACCELERATOR_MAX_DEVICE_POINTS = 60
ACCELERATOR_MIN_BUCKET_MS = 30_000
ACCELERATOR_MAX_CLUSTERS = 16
ACCELERATOR_MAX_FLEET_ROWS = 25_000
ACCELERATOR_MAX_DEVICE_ROWS = 500_000
ACCELERATOR_MAX_ATTRIBUTION_ROWS = 50_000
ACCELERATOR_MAX_RESULT_ROWS = 500_000


def accelerator_overview_dataset(
    clusters: tuple[str, ...], start_ms: int, end_ms: int, requested_bucket_ms: int, view: str | None = None
) -> DashboardDataset:
    """Build fleet, per-device, and run-attribution GPU sources."""
    validate_values("clusters", clusters, max_values=ACCELERATOR_MAX_CLUSTERS, max_length=128)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=ACCELERATOR_MAX_WINDOW_MS,
        max_window_error="Accelerator overview range must not exceed 24 hours",
        min_bucket_ms=ACCELERATOR_MIN_BUCKET_MS,
        max_points=ACCELERATOR_MAX_POINTS,
    )
    bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"
    device_bucket_ms = max(bucket_ms, math.ceil((end_ms - start_ms) / ACCELERATOR_MAX_DEVICE_POINTS))
    device_bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {device_bucket_ms}"
    cluster_values = sql_values(clusters)
    common_filter = f"""
service = 'iris-node-agent'
AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
AND timestamp_ms >= {start_ms}
AND timestamp_ms < {end_ms}
""".strip()
    if view in ("power", "gpus", "nodes"):
        # These tiles need one latest power sample per device, not device histories
        # or training attribution. Keep the full selected window for stale devices.
        sql = f"""
SELECT SUM(value) / 1000.0 AS power, COUNT(DISTINCT gpu_uuid) AS gpus, COUNT(DISTINCT node) AS nodes
FROM (
    SELECT json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           FIRST_VALUE(node_name ORDER BY timestamp_ms DESC) AS node,
           FIRST_VALUE(value ORDER BY timestamp_ms DESC) AS value
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_power_watts'
    GROUP BY cluster, json_get(attributes_json, 'gpu_uuid')
)
""".strip()
        return DashboardDataset(
            name="Accelerator current power",
            cache_key=(clusters, start_ms, end_ms),
            sources=(SourceQuery("latest_power", sql, 1),),
            setup_sql=(),
            views={name: f"SELECT {name} AS value FROM latest_power" for name in ("power", "gpus", "nodes")},
            max_result_rows=3,
        )
    if view == "faulted":
        sql = f"""
SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS "cluster", node_name AS node,
       json_get(attributes_json, 'gpu_index') AS gpu,
       MAX(CASE WHEN name = 'gpu_xid_error_code' THEN value END) AS xid,
       MAX(CASE WHEN name = 'gpu_row_remap_failures' THEN value END) AS remap_failures,
       MAX(CASE WHEN name = 'gpu_pcie_replay_errors' THEN value END) AS pcie_replays
FROM "telemetry_v1.node_agent"
WHERE {common_filter}
  AND name IN ('gpu_xid_error_code', 'gpu_row_remap_failures', 'gpu_pcie_replay_errors')
GROUP BY 1, 2, 3 HAVING MAX(value) > 0
ORDER BY xid DESC, remap_failures DESC, pcie_replays DESC
LIMIT {ACCELERATOR_MAX_DEVICE_ROWS + 1}
""".strip()
        return DashboardDataset(
            name="Accelerator faults",
            cache_key=(clusters, start_ms, end_ms),
            sources=(SourceQuery("faulted", sql, ACCELERATOR_MAX_DEVICE_ROWS),),
            setup_sql=(),
            views={"faulted": "SELECT * FROM faulted ORDER BY xid DESC, remap_failures DESC, pcie_replays DESC"},
            max_result_rows=ACCELERATOR_MAX_DEVICE_ROWS,
        )
    if view in ("power_series", "utilization_series"):
        # Sum per-device means for power; utilization remains sample-weighted.
        sql = f"""
SELECT t, origin_cluster AS series, SUM(value) / 1000.0 AS value, 'power_series' AS metric
FROM (
    SELECT {bucket} AS t, COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           node_name, json_get(attributes_json, 'gpu_uuid') AS gpu_uuid, AVG(value) AS value
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_power_watts'
    GROUP BY 1, 2, 3, 4
)
GROUP BY 1, 2
UNION ALL
SELECT {bucket} AS t, COALESCE(NULLIF(cluster, ''), 'marin') AS series,
       AVG(value) AS value, 'utilization_series' AS metric
FROM "telemetry_v1.node_agent"
WHERE {common_filter} AND name = 'gpu_utilization_percent'
GROUP BY 1, 2
ORDER BY t, series, metric
LIMIT {ACCELERATOR_MAX_FLEET_ROWS + 1}
""".strip()
        return DashboardDataset(
            name="Accelerator power and utilization",
            cache_key=(clusters, start_ms, end_ms, bucket_ms),
            sources=(SourceQuery("history", sql, ACCELERATOR_MAX_FLEET_ROWS),),
            setup_sql=(),
            views={
                name: f"SELECT t, series, value FROM history WHERE metric = '{name}' ORDER BY t, series"
                for name in ("power_series", "utilization_series")
            },
            max_result_rows=ACCELERATOR_MAX_FLEET_ROWS,
        )
    if view in ("mean_utilization", "mean_tensor", "hottest"):
        # Exact metric predicates select the existing covering projections.
        sql = f"""
SELECT utilization.value AS mean_utilization, tensor.value AS mean_tensor, temperature.value AS hottest
FROM (
    SELECT AVG(value) AS value FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_utilization_percent'
) AS utilization
CROSS JOIN (
    SELECT AVG(value) AS value FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_tensor_active_ratio'
) AS tensor
CROSS JOIN (
    SELECT MAX(value) AS value FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_temperature_celsius'
) AS temperature
""".strip()
        return DashboardDataset(
            name="Accelerator activity stats",
            cache_key=(clusters, start_ms, end_ms),
            sources=(SourceQuery("activity", sql, 1),),
            setup_sql=(),
            views={
                name: f"SELECT {name} AS value FROM activity" for name in ("mean_utilization", "mean_tensor", "hottest")
            },
            max_result_rows=3,
        )
    if view in ("tensor", "hbm", "temperature_series", "fault_count"):
        # Keep each branch within one physical metric projection. Power and
        # utilization histories are already provided by their shared source.
        sql = f"""
WITH memory_per_device AS (
    SELECT {bucket} AS t, COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           node_name, json_get(attributes_json, 'gpu_uuid') AS gpu_uuid, AVG(value) AS value
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_memory_used_bytes'
    GROUP BY 1, 2, 3, 4
), faults_per_cluster AS (
    SELECT {bucket} AS t, COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster, name,
           COUNT(DISTINCT CASE WHEN value > 0 THEN json_get(attributes_json, 'gpu_uuid') END) AS value
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter}
      AND name IN ('gpu_xid_error_code', 'gpu_row_remap_failures', 'gpu_pcie_replay_errors')
    GROUP BY 1, 2, 3
)
SELECT t, origin_cluster AS series, SUM(value) AS value, 'hbm' AS metric
FROM memory_per_device GROUP BY 1, 2
UNION ALL
SELECT {bucket} AS t, COALESCE(NULLIF(cluster, ''), 'marin') AS series,
       AVG(value) AS value, 'tensor' AS metric
FROM "telemetry_v1.node_agent"
WHERE {common_filter} AND name = 'gpu_tensor_active_ratio'
GROUP BY 1, 2
UNION ALL
SELECT {bucket} AS t, COALESCE(NULLIF(cluster, ''), 'marin') AS series,
       MAX(value) AS value, 'temperature_series' AS metric
FROM "telemetry_v1.node_agent"
WHERE {common_filter} AND name = 'gpu_temperature_celsius'
GROUP BY 1, 2
UNION ALL
SELECT t, CASE name WHEN 'gpu_xid_error_code' THEN 'XID error'
                   WHEN 'gpu_row_remap_failures' THEN 'row remap failure'
                   ELSE 'PCIe replay' END AS series,
       SUM(value) AS value, 'fault_count' AS metric
FROM faults_per_cluster GROUP BY 1, 2
ORDER BY t, metric, series
LIMIT {ACCELERATOR_MAX_FLEET_ROWS + 1}
""".strip()
        return DashboardDataset(
            name="Accelerator fleet charts",
            cache_key=(clusters, start_ms, end_ms, bucket_ms),
            sources=(SourceQuery("fleet_charts", sql, ACCELERATOR_MAX_FLEET_ROWS),),
            setup_sql=(),
            views={
                name: f"SELECT t, series, value FROM fleet_charts WHERE metric = '{name}' ORDER BY t, series"
                for name in ("tensor", "hbm", "temperature_series", "fault_count")
            },
            max_result_rows=ACCELERATOR_MAX_FLEET_ROWS,
        )
    fleet_sql = f"""
WITH raw AS (
    SELECT {bucket} AS t,
           COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           node_name,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           name, value, timestamp_ms
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter}
      AND name IN ('gpu_power_watts', 'gpu_utilization_percent', 'gpu_tensor_active_ratio',
                   'gpu_memory_used_bytes', 'gpu_temperature_celsius', 'gpu_xid_error_code',
                   'gpu_row_remap_failures', 'gpu_pcie_replay_errors')
), per_device AS (
    SELECT t, origin_cluster, node_name, gpu_uuid, name,
           AVG(value) AS value, MAX(timestamp_ms) AS last_ms
    FROM raw WHERE name IN ('gpu_power_watts', 'gpu_memory_used_bytes')
    GROUP BY 1, 2, 3, 4, 5
), additive AS (
    SELECT t, origin_cluster AS "cluster", name,
           SUM(value) AS sum_value, CAST(1 AS BIGINT) AS sample_count,
           MAX(value) AS max_value, COUNT(DISTINCT node_name) AS nodes,
           COUNT(DISTINCT gpu_uuid) AS gpus, CAST(0 AS BIGINT) AS faulty_gpus,
           MAX(last_ms) AS last_ms
    FROM per_device GROUP BY 1, 2, 3
), other AS (
    SELECT t, origin_cluster AS "cluster", name,
           SUM(value) AS sum_value, COUNT(value) AS sample_count,
           MAX(value) AS max_value, COUNT(DISTINCT node_name) AS nodes,
           COUNT(DISTINCT gpu_uuid) AS gpus,
           COUNT(DISTINCT CASE WHEN value > 0 THEN gpu_uuid END) AS faulty_gpus,
           MAX(timestamp_ms) AS last_ms
    FROM raw WHERE name NOT IN ('gpu_power_watts', 'gpu_memory_used_bytes')
    GROUP BY 1, 2, 3
)
SELECT * FROM additive
UNION ALL SELECT * FROM other
ORDER BY t, "cluster", name
LIMIT {ACCELERATOR_MAX_FLEET_ROWS + 1}
""".strip()
    inventory_start_ms = max(start_ms, end_ms - 10 * 60 * 1000)
    # Fault tables have a separate panel window. Only the combined dataset needs
    # their rows alongside device histories and freshness.
    fault_union = "UNION ALL SELECT * FROM faults" if view is None else ""
    latest_sql = f"""
    SELECT 'latest' AS kind,
           CAST(NULL AS BIGINT) AS t,
           COALESCE(NULLIF(cluster, ''), 'marin') AS "cluster",
           node_name AS node,
           json_get(attributes_json, 'gpu_index') AS gpu,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           name,
           value,
           json_get(attributes_json, 'gpu_model') AS series,
           timestamp_ms AS last_ms,
           ROW_NUMBER() OVER (
               PARTITION BY cluster, json_get(attributes_json, 'gpu_uuid'), name
               ORDER BY timestamp_ms DESC
           ) AS rn
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter}
      AND name IN ('gpu_power_watts', 'hardware_inventory')
""".strip()
    if view is not None:
        # Freshness only consumes latest power rows. Ordered aggregates avoid
        # sorting each device's history and select its latest node after moves.
        latest_sql = f"""
    SELECT 'latest' AS kind, CAST(NULL AS BIGINT) AS t,
           COALESCE(NULLIF(cluster, ''), 'marin') AS "cluster",
           FIRST_VALUE(node_name ORDER BY timestamp_ms DESC) AS node,
           FIRST_VALUE(json_get(attributes_json, 'gpu_index') ORDER BY timestamp_ms DESC) AS gpu,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid, name,
           FIRST_VALUE(value ORDER BY timestamp_ms DESC) AS value,
           FIRST_VALUE(json_get(attributes_json, 'gpu_model') ORDER BY timestamp_ms DESC) AS series,
           MAX(timestamp_ms) AS last_ms, 1 AS rn
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_power_watts'
    GROUP BY cluster, json_get(attributes_json, 'gpu_uuid'), name
""".strip()
    device_sql = f"""
WITH latest AS (
{latest_sql}
), faults AS (
    SELECT 'fault' AS kind,
           CAST(NULL AS BIGINT) AS t,
           COALESCE(NULLIF(cluster, ''), 'marin') AS "cluster",
           node_name AS node,
           json_get(attributes_json, 'gpu_index') AS gpu,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           name,
           MAX(value) AS value,
           CAST(NULL AS VARCHAR) AS series,
           MAX(timestamp_ms) AS last_ms,
           1 AS rn
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter}
      AND name IN ('gpu_xid_error_code', 'gpu_row_remap_failures', 'gpu_pcie_replay_errors')
    GROUP BY 3, 4, 5, 6, 7
), sm AS (
    SELECT 'sm' AS kind,
           {device_bucket} AS t,
           COALESCE(NULLIF(cluster, ''), 'marin') AS "cluster",
           node_name AS node,
           json_get(attributes_json, 'gpu_index') AS gpu,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           name,
           AVG(value) * 100.0 AS value,
           CAST(NULL AS VARCHAR) AS series,
           MAX(timestamp_ms) AS last_ms,
           1 AS rn
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_sm_active_ratio'
    GROUP BY 2, 3, 4, 5, 6, 7
), temperature_per_gpu AS (
    SELECT {device_bucket} AS t,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           AVG(value) AS sample,
           MAX(timestamp_ms) AS last_ms
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_temperature_celsius'
    GROUP BY 1, 2
), temperature_distribution AS (
    SELECT 'temperature_distribution' AS kind,
           t,
           CAST(NULL AS VARCHAR) AS "cluster",
           CAST(NULL AS VARCHAR) AS node,
           CAST(NULL AS VARCHAR) AS gpu,
           CAST(NULL AS VARCHAR) AS gpu_uuid,
           'gpu_temperature_celsius' AS name,
           CAST(COUNT(*) AS DOUBLE) AS value,
           CAST(CEIL(sample / 5.0) * 5.0 AS VARCHAR) AS series,
           MAX(last_ms) AS last_ms,
           1 AS rn
    FROM temperature_per_gpu
    GROUP BY 2, 9
), model AS (
    SELECT json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           MAX(json_get(attributes_json, 'gpu_model')) AS gpu_model
    FROM "telemetry_v1.node_agent"
    WHERE service = 'iris-node-agent'
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND timestamp_ms >= {inventory_start_ms}
      AND timestamp_ms < {end_ms}
      AND name = 'hardware_inventory'
    GROUP BY 1
), power_per_gpu AS (
    SELECT {start_ms} + (timestamp_ms - {start_ms})
               - (timestamp_ms - {start_ms}) % {bucket_ms} AS t,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           AVG(value) AS watts,
           MAX(timestamp_ms) AS last_ms
    FROM "telemetry_v1.node_agent"
    WHERE service = 'iris-node-agent'
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND timestamp_ms >= {start_ms}
      AND timestamp_ms < {end_ms}
      AND name = 'gpu_power_watts'
    GROUP BY 1, 2
), model_power AS (
    SELECT 'model_power' AS kind,
           power.t,
           CAST(NULL AS VARCHAR) AS "cluster",
           CAST(NULL AS VARCHAR) AS node,
           CAST(NULL AS VARCHAR) AS gpu,
           CAST(NULL AS VARCHAR) AS gpu_uuid,
           'gpu_power_watts' AS name,
           SUM(power.watts) / 1000.0 AS value,
           COALESCE(model.gpu_model, 'unknown') AS series,
           MAX(power.last_ms) AS last_ms,
           1 AS rn
    FROM power_per_gpu AS power
    LEFT JOIN model USING (gpu_uuid)
    GROUP BY 2, 9
)
SELECT kind, t, "cluster", node, gpu, gpu_uuid, name, value, series, last_ms
FROM (
    SELECT * FROM latest
    {fault_union}
    UNION ALL SELECT * FROM sm
    UNION ALL SELECT * FROM temperature_distribution
    UNION ALL SELECT * FROM model_power
) WHERE rn = 1
ORDER BY kind, t, "cluster", node, gpu
LIMIT {ACCELERATOR_MAX_DEVICE_ROWS + 1}
""".strip()
    attribution_sql = f"""
WITH gpu AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           {bucket} AS t,
           node_name AS node,
           json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
           AVG(value) AS watts
    FROM "telemetry_v1.node_agent"
    WHERE {common_filter} AND name = 'gpu_power_watts'
    GROUP BY 1, 2, 3, 4
), run_raw AS (
    SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
           {bucket} AS t,
           node_name AS node,
           run_id AS run
    FROM "levanter.metrics"
    WHERE step IS NOT NULL
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({cluster_values})
      AND timestamp_ms >= {start_ms}
      AND timestamp_ms < {end_ms}
), run_node AS (
    SELECT origin_cluster, t, node, run
    FROM (
        SELECT origin_cluster, t, node, run,
               ROW_NUMBER() OVER (PARTITION BY origin_cluster, t, node ORDER BY COUNT(*) DESC, run) AS rn
        FROM run_raw GROUP BY 1, 2, 3, 4
    ) WHERE rn = 1
)
SELECT gpu.t, COALESCE(run_node.run, '(idle / unattributed)') AS series, SUM(gpu.watts) / 1000.0 AS value
FROM gpu LEFT JOIN run_node USING (origin_cluster, t, node)
GROUP BY 1, 2 ORDER BY 1, 2
LIMIT {ACCELERATOR_MAX_ATTRIBUTION_ROWS + 1}
""".strip()
    views = {
        "power": "SELECT SUM(value) / 1000.0 AS value FROM devices WHERE kind = 'latest' AND name = 'gpu_power_watts'",
        "mean_utilization": (
            "SELECT SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value "
            "FROM fleet WHERE name = 'gpu_utilization_percent'"
        ),
        "mean_tensor": (
            "SELECT SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value "
            "FROM fleet WHERE name = 'gpu_tensor_active_ratio'"
        ),
        "hottest": "SELECT MAX(max_value) AS value FROM fleet WHERE name = 'gpu_temperature_celsius'",
        "nodes": "SELECT COUNT(DISTINCT node) AS value FROM devices WHERE kind = 'latest' AND name = 'gpu_power_watts'",
        "gpus": (
            "SELECT COUNT(DISTINCT gpu_uuid) AS value FROM devices WHERE kind = 'latest' AND name = 'gpu_power_watts'"
        ),
        "power_series": (
            "SELECT t, cluster AS series, SUM(sum_value) / 1000.0 AS value "
            "FROM fleet WHERE name = 'gpu_power_watts' GROUP BY 1, 2 ORDER BY 1"
        ),
        "utilization_series": (
            "SELECT t, cluster AS series, SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value "
            "FROM fleet WHERE name = 'gpu_utilization_percent' GROUP BY 1, 2 ORDER BY 1"
        ),
        "run_power": "SELECT * FROM attribution ORDER BY t, series",
        "tensor": (
            "SELECT t, cluster AS series, SUM(sum_value) / NULLIF(SUM(sample_count), 0) AS value "
            "FROM fleet WHERE name = 'gpu_tensor_active_ratio' GROUP BY 1, 2 ORDER BY 1"
        ),
        "hbm": (
            "SELECT t, cluster AS series, SUM(sum_value) AS value "
            "FROM fleet WHERE name = 'gpu_memory_used_bytes' GROUP BY 1, 2 ORDER BY 1"
        ),
        "temperature_series": (
            "SELECT t, cluster AS series, MAX(max_value) AS value "
            "FROM fleet WHERE name = 'gpu_temperature_celsius' GROUP BY 1, 2 ORDER BY 1"
        ),
        "fault_count": (
            """
SELECT t,
       CASE name WHEN 'gpu_xid_error_code' THEN 'XID error'
                 WHEN 'gpu_row_remap_failures' THEN 'row remap failure'
                 ELSE 'PCIe replay' END AS series,
       SUM(faulty_gpus) AS value
FROM fleet
WHERE name IN ('gpu_xid_error_code', 'gpu_row_remap_failures', 'gpu_pcie_replay_errors')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "faulted": (
            """
SELECT cluster, node, gpu,
       MAX(CASE WHEN name = 'gpu_xid_error_code' THEN value END) AS xid,
       MAX(CASE WHEN name = 'gpu_row_remap_failures' THEN value END) AS remap_failures,
       MAX(CASE WHEN name = 'gpu_pcie_replay_errors' THEN value END) AS pcie_replays
FROM devices WHERE kind = 'fault'
GROUP BY 1, 2, 3 HAVING MAX(value) > 0 ORDER BY xid DESC, remap_failures DESC, pcie_replays DESC
""".strip()
        ),
        "freshness": (
            f"""
SELECT cluster, COUNT(DISTINCT gpu_uuid) AS gpus, COUNT(DISTINCT node) AS nodes,
       ({end_ms} - MAX(last_ms)) / 1000.0 AS lag_seconds
FROM devices WHERE kind = 'latest' AND name = 'gpu_power_watts'
GROUP BY 1 ORDER BY lag_seconds DESC
""".strip()
        ),
        "model_power": (
            "SELECT t, series, SUM(value) AS value FROM devices WHERE kind = 'model_power' GROUP BY 1, 2 ORDER BY 1"
        ),
        "sm": (
            "SELECT t, cluster, node, gpu, value AS sm_utilization "
            "FROM devices WHERE kind = 'sm' ORDER BY cluster, node, gpu, t"
        ),
        "temperature_distribution": (
            "SELECT t, series, SUM(value) AS value FROM devices "
            "WHERE kind = 'temperature_distribution' GROUP BY 1, 2 ORDER BY 1, 2"
        ),
    }
    sources = (
        SourceQuery("fleet", fleet_sql, ACCELERATOR_MAX_FLEET_ROWS),
        SourceQuery("devices", device_sql, ACCELERATOR_MAX_DEVICE_ROWS),
        SourceQuery("attribution", attribution_sql, ACCELERATOR_MAX_ATTRIBUTION_ROWS),
    )
    source_views = {
        "devices": ("freshness", "model_power", "sm", "temperature_distribution"),
        "attribution": ("run_power",),
    }
    setup_sql = ()
    if view is not None:
        if view not in views:
            raise ValueError(f"unknown Accelerator overview view {view!r}")
        for source_name, names in source_views.items():
            if view in names:
                sources = tuple(source for source in sources if source.name == source_name)
                views = {name: views[name] for name in names}
                break
        if view in source_views["devices"]:
            # Device rasters repeat labels for every point. Bound the samples
            # before packing, then enforce that same budget on the Arrow lists.
            packed_sql = f"""
SELECT kind, "cluster", node, gpu, gpu_uuid, name, series,
       array_agg(named_struct('t', t, 'value', value, 'last_ms', last_ms)) AS points
FROM ({device_sql}) AS device_samples
GROUP BY 1, 2, 3, 4, 5, 6, 7
LIMIT {ACCELERATOR_MAX_DEVICE_ROWS + 1}
""".strip()
            sources = (
                SourceQuery("device_series", packed_sql, ACCELERATOR_MAX_DEVICE_ROWS, ACCELERATOR_MAX_DEVICE_ROWS),
            )
            setup_sql = (
                """
CREATE VIEW devices AS
SELECT kind, point.t AS t, "cluster", node, gpu, gpu_uuid, name,
       point.value AS value, series, point.last_ms AS last_ms
FROM (SELECT * EXCLUDE (points), unnest(points) AS point FROM device_series)
""".strip(),
            )
    return DashboardDataset(
        name="Accelerator overview",
        cache_key=(clusters, start_ms, end_ms, bucket_ms, tuple(source.name for source in sources)),
        sources=sources,
        setup_sql=setup_sql,
        views=views,
        max_result_rows=ACCELERATOR_MAX_RESULT_ROWS,
    )
