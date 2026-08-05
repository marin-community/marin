import type { DashboardDefinition, DashboardPanel, DashboardPanelWidth } from '@/types/dashboard'

const COUNTER_LOOKBACK_MS = 300_000

function trainingMetricPanel(
  id: string,
  title: string,
  description: string,
  metricName: string,
  series: string,
  width: DashboardPanelWidth,
): DashboardPanel {
  return {
    id,
    title,
    kind: 'timeseries',
    width,
    description,
    sql: `SELECT date_bin(INTERVAL '{{interval_ms}} milliseconds', to_timestamp_millis(timestamp_ms)) AS t,
       '${series}' AS series,
       AVG(value) AS value
FROM telemetry_v1
WHERE service = 'levanter'
  AND name = '${metricName}'
  AND json_get(resource_attributes_json, 'job_id') = {{job_id}}
  AND timestamp_ms >= {{from_ms}}
  AND timestamp_ms < {{to_ms}}
GROUP BY 1
ORDER BY 1`,
  }
}

const HARDWARE_AND_COMMUNICATIONS: DashboardDefinition = {
  version: 1,
  id: 'hardware-communications',
  title: 'Hardware and communications',
  description: 'GPU/DCGM faults, counter deltas, and a job-node network proxy for NCCL traffic.',
  variables: [{ name: 'job_id', label: 'Iris job ID', default: '' }],
  panels: [
    {
      id: 'ras-state',
      title: 'Current RAS and communicator faults',
      kind: 'table',
      width: 'full',
      description: 'Latest nonzero device state in the selected range. NCCL RAS rows appear when that probe is emitted.',
      sql: `WITH ranked AS (
  SELECT to_timestamp_millis(timestamp_ms) AS observed_at,
         COALESCE(json_get(attributes_json, 'node_name'),
                  json_get(resource_attributes_json, 'node_name'), 'unknown') AS node,
         json_get(attributes_json, 'gpu_uuid') AS gpu_uuid,
         name,
         value,
         "cluster" AS origin_cluster,
         attributes_json,
         ROW_NUMBER() OVER (
           PARTITION BY "cluster", name, resource_attributes_json, attributes_json
           ORDER BY timestamp_ms DESC, seq DESC
         ) AS recency
  FROM telemetry_v1
  WHERE timestamp_ms >= {{from_ms}}
    AND timestamp_ms < {{to_ms}}
    AND name IN (
      'gpu_xid_error_code',
      'gpu_row_remap_failures',
      'gpu_health_status',
      'gpu_pcie_replay_errors',
      'gpu_nvlink_errors',
      'communicator_state',
      'communicator_rank_status'
    )
)
SELECT observed_at, origin_cluster, node, gpu_uuid, name, value, attributes_json
FROM ranked
WHERE recency = 1 AND value <> 0
ORDER BY observed_at DESC
LIMIT 500`,
    },
    {
      id: 'error-deltas',
      title: 'GPU link and PCIe error deltas',
      kind: 'timeseries',
      width: 'half',
      description: 'Positive increments from cumulative DCGM snapshots; counter resets are discarded.',
      sql: `WITH samples AS (
  SELECT timestamp_ms,
         seq,
         name,
         value,
         "cluster" AS origin_cluster,
         resource_attributes_json,
         attributes_json,
         LAG(value) OVER (
           PARTITION BY "cluster", name, resource_attributes_json, attributes_json
           ORDER BY timestamp_ms, seq
         ) AS previous_value
  FROM telemetry_v1
  WHERE timestamp_ms >= {{from_ms}} - ${COUNTER_LOOKBACK_MS}
    AND timestamp_ms < {{to_ms}}
    AND name IN ('gpu_pcie_replay_errors', 'gpu_nvlink_errors')
), increments AS (
  SELECT timestamp_ms,
         name,
         CASE WHEN previous_value IS NULL OR value < previous_value
              THEN NULL ELSE value - previous_value END AS delta
  FROM samples
)
SELECT date_bin(INTERVAL '{{interval_ms}} milliseconds', to_timestamp_millis(timestamp_ms)) AS t,
       name AS series,
       SUM(delta) AS value
FROM increments
WHERE timestamp_ms >= {{from_ms}} AND delta > 0
GROUP BY 1, 2
ORDER BY 1, 2`,
    },
    {
      id: 'job-network',
      title: 'Job-node network throughput (NCCL proxy)',
      kind: 'timeseries',
      width: 'half',
      description: 'Aggregate physical-interface traffic on nodes observed for the job. Includes non-NCCL and colocated traffic.',
      sql: `WITH job_nodes AS (
  SELECT DISTINCT json_get(resource_attributes_json, 'node_name') AS node_name
  FROM telemetry_v1
  WHERE service = 'levanter'
    AND timestamp_ms >= {{from_ms}}
    AND timestamp_ms < {{to_ms}}
    AND json_get(resource_attributes_json, 'job_id') = {{job_id}}
    AND json_get(resource_attributes_json, 'node_name') IS NOT NULL
), samples AS (
  SELECT timestamp_ms,
         seq,
         name,
         value,
         COALESCE(json_get(attributes_json, 'node_name'),
                  json_get(resource_attributes_json, 'node_name')) AS node_name,
         LAG(value) OVER (
           PARTITION BY "cluster", name, resource_attributes_json, attributes_json
           ORDER BY timestamp_ms, seq
         ) AS previous_value,
         LAG(timestamp_ms) OVER (
           PARTITION BY "cluster", name, resource_attributes_json, attributes_json
           ORDER BY timestamp_ms, seq
         ) AS previous_timestamp_ms
  FROM telemetry_v1
  WHERE service = 'node'
    AND timestamp_ms >= {{from_ms}} - ${COUNTER_LOOKBACK_MS}
    AND timestamp_ms < {{to_ms}}
    AND name IN ('node_network_receive_bytes', 'node_network_transmit_bytes')
), node_bins AS (
  SELECT date_bin(INTERVAL '{{interval_ms}} milliseconds', to_timestamp_millis(timestamp_ms)) AS t,
         node_name,
         CASE name
           WHEN 'node_network_receive_bytes' THEN 'receive Gbit/s'
           ELSE 'transmit Gbit/s'
         END AS series,
         AVG((value - previous_value) * 8000.0 / (timestamp_ms - previous_timestamp_ms) / 1000000000.0) AS value
  FROM samples
  WHERE timestamp_ms >= {{from_ms}}
    AND node_name IN (SELECT node_name FROM job_nodes)
    AND previous_value IS NOT NULL
    AND value >= previous_value
    AND timestamp_ms > previous_timestamp_ms
  GROUP BY 1, 2, 3
)
SELECT t, series, SUM(value) AS value
FROM node_bins
GROUP BY 1, 2
ORDER BY 1, 2`,
    },
  ],
}

const TRAINING_RUN: DashboardDefinition = {
  version: 1,
  id: 'training-run',
  title: 'Training run',
  description: 'A deliberately small view of Levanter progress, throughput, and utilization.',
  variables: [{ name: 'job_id', label: 'Iris job ID', default: '' }],
  panels: [
    trainingMetricPanel(
      'throughput',
      'Training throughput',
      'Mean tokens per second across workers reporting for the job.',
      'throughput_tokens_per_second',
      'tokens/s',
      'half',
    ),
    trainingMetricPanel(
      'mfu',
      'Mean MFU',
      'Mean model-flop utilization percentage across workers.',
      'throughput_mean_mfu',
      'MFU %',
      'half',
    ),
    trainingMetricPanel(
      'loss',
      'Training loss',
      'Mean train/loss snapshots for the selected job.',
      'train_loss',
      'train loss',
      'full',
    ),
  ],
}

const INFERENCE_SERVE: DashboardDefinition = {
  version: 1,
  id: 'inference-serve',
  title: 'Inference serve',
  description: 'Queue pressure, KV-cache use, and mean TTFT for one vLLM Iris job.',
  variables: [{ name: 'job_id', label: 'Iris serving job ID', default: '' }],
  panels: [
    {
      id: 'saturation',
      title: 'Queue and KV cache',
      kind: 'timeseries',
      width: 'half',
      description: 'Waiting requests are summed across replicas; KV-cache utilization is averaged.',
      sql: `WITH bins AS (
  SELECT date_bin(INTERVAL '{{interval_ms}} milliseconds', to_timestamp_millis(timestamp_ms)) AS t,
         CASE WHEN name IN ('kv_cache_usage_perc', 'gpu_cache_usage_perc')
              THEN 'KV cache %' ELSE 'waiting requests' END AS series,
         CASE WHEN name IN ('kv_cache_usage_perc', 'gpu_cache_usage_perc')
              THEN AVG(value) ELSE SUM(value) END AS value
  FROM telemetry_v1
  WHERE service = 'vllm'
    AND name IN ('num_requests_waiting', 'kv_cache_usage_perc', 'gpu_cache_usage_perc')
    AND json_get(resource_attributes_json, 'job_id') = {{job_id}}
    AND timestamp_ms >= {{from_ms}}
    AND timestamp_ms < {{to_ms}}
  GROUP BY 1, 2, name
)
SELECT t, series, AVG(value) AS value
FROM bins
GROUP BY 1, 2
ORDER BY 1, 2`,
    },
    {
      id: 'ttft',
      title: 'Mean time to first token',
      kind: 'timeseries',
      width: 'half',
      description: 'Derived from cumulative histogram sum/count snapshots; source resets are discarded.',
      sql: `WITH samples AS (
  SELECT timestamp_ms,
         seq,
         name,
         value,
         "cluster" AS origin_cluster,
         resource_attributes_json,
         attributes_json,
         LAG(value) OVER (
           PARTITION BY "cluster", name, resource_attributes_json, attributes_json
           ORDER BY timestamp_ms, seq
         ) AS previous_value
  FROM telemetry_v1
  WHERE service = 'vllm'
    AND name IN ('time_to_first_token_seconds_sum', 'time_to_first_token_seconds_count')
    AND json_get(resource_attributes_json, 'job_id') = {{job_id}}
    AND timestamp_ms >= {{from_ms}} - ${COUNTER_LOOKBACK_MS}
    AND timestamp_ms < {{to_ms}}
), deltas AS (
  SELECT timestamp_ms,
         name,
         CASE WHEN previous_value IS NULL OR value < previous_value
              THEN NULL ELSE value - previous_value END AS delta
  FROM samples
), bins AS (
  SELECT date_bin(INTERVAL '{{interval_ms}} milliseconds', to_timestamp_millis(timestamp_ms)) AS t,
         SUM(CASE WHEN name = 'time_to_first_token_seconds_sum' THEN delta ELSE 0 END) AS total_seconds,
         SUM(CASE WHEN name = 'time_to_first_token_seconds_count' THEN delta ELSE 0 END) AS request_count
  FROM deltas
  WHERE timestamp_ms >= {{from_ms}} AND delta IS NOT NULL
  GROUP BY 1
)
SELECT t, 'mean TTFT seconds' AS series, total_seconds / NULLIF(request_count, 0) AS value
FROM bins
WHERE request_count > 0
ORDER BY 1`,
    },
  ],
}

export const BUILT_IN_DASHBOARDS: DashboardDefinition[] = [
  HARDWARE_AND_COMMUNICATIONS,
  TRAINING_RUN,
  INFERENCE_SERVE,
]
