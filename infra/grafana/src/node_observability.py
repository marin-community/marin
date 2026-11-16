# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One bounded node-agent fetch shared by the Node Details panels."""

from dashboard_dataset import DashboardDataset, SourceQuery, bounded_bucket_ms, validate_values
from vllm_observability import sql_values

NODE_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
NODE_MAX_POINTS = 360
NODE_MAX_SERIES = 2_048
NODE_MAX_SAMPLES = 500_000
NODE_MAX_RESULT_ROWS = 25_000
NODE_MIN_BUCKET_MS = 15_000
NODE_MAX_CLUSTERS = 16
NODE_MAX_NODES = 32
NODE_IDENTITY_MAX_LENGTH = 256

_METRIC_NAMES = (
    "gpu_memory_temperature_celsius",
    "gpu_memory_total_bytes",
    "gpu_memory_used_bytes",
    "gpu_nvlink_receive_bytes_per_second",
    "gpu_nvlink_transmit_bytes_per_second",
    "gpu_pcie_receive_bytes_per_second",
    "gpu_pcie_replay_errors",
    "gpu_pcie_transmit_bytes_per_second",
    "gpu_power_watts",
    "gpu_row_remap_failures",
    "gpu_sm_active_ratio",
    "gpu_temperature_celsius",
    "gpu_tensor_active_ratio",
    "gpu_utilization_percent",
    "gpu_xid_error_code",
    "hardware_inventory",
    "node_cpu_utilization_percent",
    "node_disk_total_bytes",
    "node_disk_used_bytes",
    "node_memory_total_bytes",
    "node_memory_used_bytes",
    "node_network_receive_bytes",
    "node_network_transmit_bytes",
)


def node_overview_dataset(
    clusters: tuple[str, ...],
    nodes: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
) -> DashboardDataset:
    """Build the fixed Node Details source query and local projections."""
    validate_values("clusters", clusters, max_values=NODE_MAX_CLUSTERS, max_length=NODE_IDENTITY_MAX_LENGTH)
    validate_values("nodes", nodes, max_values=NODE_MAX_NODES, max_length=NODE_IDENTITY_MAX_LENGTH)
    bucket_ms = bounded_bucket_ms(
        start_ms,
        end_ms,
        requested_bucket_ms,
        max_window_ms=NODE_MAX_WINDOW_MS,
        max_window_error="node overview range must not exceed 7 days",
        min_bucket_ms=NODE_MIN_BUCKET_MS,
        max_points=NODE_MAX_POINTS,
    )
    scan_start_ms = max(0, start_ms - bucket_ms)
    source_sql = f"""
SELECT COALESCE(NULLIF(cluster, ''), 'marin') AS origin_cluster,
       node_name,
       name,
       attributes_json,
       array_agg(named_struct('timestamp_ms', timestamp_ms, 'seq', seq, 'value', value)) AS points
FROM (
    SELECT cluster, node_name, name, attributes_json, timestamp_ms, seq, value
    FROM "telemetry_v1.node_agent"
    WHERE service = 'iris-node-agent'
      AND name IN ({sql_values(_METRIC_NAMES)})
      AND node_name IN ({sql_values(nodes)})
      AND COALESCE(NULLIF(cluster, ''), 'marin') IN ({sql_values(clusters)})
      AND timestamp_ms >= {scan_start_ms}
      AND timestamp_ms < {end_ms}
    LIMIT {NODE_MAX_SAMPLES + 1}
) AS bounded_samples
GROUP BY 1, 2, 3, 4
LIMIT {NODE_MAX_SERIES + 1}
""".strip()

    setup_sql = (
        """
CREATE VIEW telemetry AS
SELECT origin_cluster AS cluster,
       node_name,
       name,
       attributes_json,
       point.timestamp_ms AS timestamp_ms,
       point.seq AS seq,
       point.value AS value
FROM (SELECT * EXCLUDE (points), unnest(points) AS point FROM node_series)
""".strip(),
    )
    bucket = f"{start_ms} + (timestamp_ms - {start_ms}) - (timestamp_ms - {start_ms}) % {bucket_ms}"
    visible = f"timestamp_ms >= {start_ms}"
    views = {
        "engine_activity": (
            f"""
SELECT {bucket} AS t,
       'GPU ' || json_get(attributes_json, 'gpu_index') || ' · ' ||
           CASE name WHEN 'gpu_utilization_percent' THEN 'utilization'
                     WHEN 'gpu_sm_active_ratio' THEN 'SM active' ELSE 'tensor active' END AS series,
       AVG(CASE WHEN name = 'gpu_utilization_percent' THEN value ELSE value * 100.0 END) AS value
FROM telemetry
WHERE {visible} AND name IN ('gpu_utilization_percent', 'gpu_sm_active_ratio', 'gpu_tensor_active_ratio')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "temperatures": (
            f"""
SELECT {bucket} AS t,
       'GPU ' || json_get(attributes_json, 'gpu_index') ||
           CASE name WHEN 'gpu_memory_temperature_celsius' THEN ' · HBM' ELSE ' · core' END AS series,
       AVG(value) AS value
FROM telemetry
WHERE {visible} AND name IN ('gpu_temperature_celsius', 'gpu_memory_temperature_celsius')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "power": (
            f"""
SELECT {bucket} AS t, 'GPU ' || json_get(attributes_json, 'gpu_index') AS series, AVG(value) AS value
FROM telemetry WHERE {visible} AND name = 'gpu_power_watts'
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "memory": (
            f"""
SELECT {bucket} AS t,
       'GPU ' || json_get(attributes_json, 'gpu_index') AS series,
       100.0 * AVG(CASE WHEN name = 'gpu_memory_used_bytes' THEN value END)
           / NULLIF(AVG(CASE WHEN name = 'gpu_memory_total_bytes' THEN value END), 0) AS value
FROM telemetry
WHERE {visible} AND name IN ('gpu_memory_used_bytes', 'gpu_memory_total_bytes')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "interconnect": (
            f"""
SELECT {bucket} AS t,
       'GPU ' || json_get(attributes_json, 'gpu_index') || ' · ' ||
           CASE name WHEN 'gpu_nvlink_receive_bytes_per_second' THEN 'NVLink RX'
                     WHEN 'gpu_nvlink_transmit_bytes_per_second' THEN 'NVLink TX'
                     WHEN 'gpu_pcie_receive_bytes_per_second' THEN 'PCIe RX' ELSE 'PCIe TX' END AS series,
       AVG(value) AS value
FROM telemetry
WHERE {visible} AND name IN ('gpu_nvlink_receive_bytes_per_second', 'gpu_nvlink_transmit_bytes_per_second',
                             'gpu_pcie_receive_bytes_per_second', 'gpu_pcie_transmit_bytes_per_second')
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "host_utilization": (
            f"""
WITH raw AS (
    SELECT {bucket} AS t, name, AVG(value) AS value
    FROM telemetry
    WHERE {visible} AND name IN ('node_cpu_utilization_percent', 'node_memory_used_bytes',
                                 'node_memory_total_bytes', 'node_disk_used_bytes', 'node_disk_total_bytes')
    GROUP BY 1, 2
)
SELECT t, 'CPU' AS series, MAX(CASE WHEN name = 'node_cpu_utilization_percent' THEN value END) AS value
FROM raw GROUP BY 1
UNION ALL
SELECT t, 'memory', 100.0 * MAX(CASE WHEN name = 'node_memory_used_bytes' THEN value END)
    / NULLIF(MAX(CASE WHEN name = 'node_memory_total_bytes' THEN value END), 0)
FROM raw GROUP BY 1
UNION ALL
SELECT t, 'local disk', 100.0 * MAX(CASE WHEN name = 'node_disk_used_bytes' THEN value END)
    / NULLIF(MAX(CASE WHEN name = 'node_disk_total_bytes' THEN value END), 0)
FROM raw GROUP BY 1 ORDER BY 1
""".strip()
        ),
        "host_network": (
            f"""
WITH samples AS (
    SELECT timestamp_ms, name, value,
           LAG(value) OVER (PARTITION BY cluster, node_name, name, json_get(attributes_json, 'source_replica_uid')
                            ORDER BY timestamp_ms, seq) AS prior_value
    FROM telemetry
    WHERE name IN ('node_network_receive_bytes', 'node_network_transmit_bytes')
)
SELECT {bucket} AS t,
       CASE name WHEN 'node_network_receive_bytes' THEN 'receive' ELSE 'transmit' END AS series,
       SUM(CASE WHEN value >= prior_value THEN value - prior_value ELSE 0 END) * 1000.0 / {bucket_ms} AS value
FROM samples WHERE {visible}
GROUP BY 1, 2 ORDER BY 1
""".strip()
        ),
        "inventory": (
            f"""
SELECT json_get(attributes_json, 'gpu_index') AS gpu,
       json_get(attributes_json, 'gpu_uuid') AS uuid,
       json_get(attributes_json, 'pci_bus_id') AS pci_bus,
       MAX(json_get(attributes_json, 'gpu_model')) AS model,
       MAX(json_get(attributes_json, 'driver_version')) AS driver,
       ({end_ms} - MAX(timestamp_ms)) / 1000.0 AS lag
FROM telemetry
WHERE {visible} AND name = 'hardware_inventory' AND json_get(attributes_json, 'device_kind') = 'gpu'
GROUP BY 1, 2, 3 ORDER BY 1
""".strip()
        ),
        "faults": (
            f"""
SELECT json_get(attributes_json, 'gpu_index') AS gpu,
       MAX(CASE WHEN name = 'gpu_xid_error_code' THEN value END) AS xid,
       MAX(CASE WHEN name = 'gpu_row_remap_failures' THEN value END) AS remap_failures,
       MAX(CASE WHEN name = 'gpu_pcie_replay_errors' THEN value END) AS pcie_replays
FROM telemetry
WHERE {visible} AND name IN ('gpu_xid_error_code', 'gpu_row_remap_failures', 'gpu_pcie_replay_errors')
GROUP BY 1 HAVING MAX(value) > 0 ORDER BY 2 DESC, 3 DESC, 4 DESC
""".strip()
        ),
    }
    return DashboardDataset(
        name="node overview",
        cache_key=(clusters, nodes, start_ms, end_ms, bucket_ms),
        sources=(SourceQuery("node_series", source_sql, NODE_MAX_SERIES, NODE_MAX_SAMPLES),),
        setup_sql=setup_sql,
        views=views,
        max_result_rows=NODE_MAX_RESULT_ROWS,
    )
