// Synthetic-canary view, read from the finelog metrics the infra/probes
// daemon writes (namespace `infra.canary.metrics`).
//
// Every probe cycle emits flat `Sample` rows — { metric, value, labels (a
// JSON object string), collected_at } — covering both up/down health checks
// and numeric gauges (see infra/probes/src/sample.py). This module turns two
// bounded DuckDB queries over that namespace into a typed snapshot:
//
//   - checks: the latest probe_up / probe_latency_ms per synthetic health
//     check (controller-ping, finelog-write, iris-job-submit/<zone>).
//   - provisioning: the latest accelerator-provisioning cycle the probes roll
//     up from the controller's iris.provisioning namespace — fleet success
//     ratio, create→ready latency, outcome counts, and a per-pool breakdown
//     (see infra/probes/src/provisioning.py for the metric vocabulary).

import { queryFinelog, sqlTimestampUtc } from "./finelogQuery.js";

const METRICS_NAMESPACE = "infra.canary.metrics";

// How far back each query scans for "latest" rows. The health checks run on a
// ≤5min cadence and provisioning every 15min, so two hours comfortably covers
// the freshest cycle while bounding the scan; a longer reach only matters
// right after a canary restart, where MAX(collected_at) still wins.
const CHECKS_LOOKBACK_MS = 2 * 60 * 60 * 1000;
// Provisioning rolls a trailing 3h window forward every 15min; reach back far
// enough that a brief canary outage doesn't blank the panel.
const PROVISIONING_LOOKBACK_MS = 6 * 60 * 60 * 1000;

// Metric names emitted by the probe collectors (infra/probes/src). Kept in
// sync with runner.py (health meta-metrics) and provisioning.py (gauges).
const METRIC_UP = "probe_up";
const METRIC_LATENCY_MS = "probe_latency_ms";
const PROVISION_PREFIX = "provision_";
const METRIC_PROVISION_READY = "provision_ready";
const METRIC_PROVISION_STOCKOUT = "provision_stockout";
const METRIC_PROVISION_ERROR = "provision_error";
const METRIC_PROVISION_PREEMPTED = "provision_preempted";
const METRIC_PROVISION_OUTCOMES = "provision_outcomes";
const METRIC_PROVISION_SUCCESS_RATIO = "provision_success_ratio";
const METRIC_PROVISION_POOLS_PLACING = "provision_pools_placing";
const METRIC_PROVISION_POOLS_STOCKOUT_DEAD = "provision_pools_stockout_dead";
const METRIC_PROVISION_WINDOW_HOURS = "provision_window_hours";
const METRIC_PROVISION_LATENCY_SECONDS = "provision_latency_seconds";

const FLEET_SCOPE = "fleet";

export interface ProbeCheck {
  probe: string;
  up: boolean;
  latencyMs: number | null;
  collectedAt: string; // ISO of the latest probe_up sample
}

export interface ProvisionPool {
  resourceType: string;
  scaleGroup: string;
  zone: string;
  ready: number;
  stockout: number;
  error: number;
  preempted: number;
  outcomes: number; // resolved create attempts = ready + stockout + error
  successRatio: number | null; // ready / outcomes
  latencyP50Seconds: number | null;
  latencyP95Seconds: number | null;
}

export interface ProvisionFleet {
  ready: number;
  stockout: number;
  error: number;
  preempted: number;
  outcomes: number;
  successRatio: number | null;
  poolsPlacing: number;
  poolsStockoutDead: number;
  latencyP50Seconds: number | null;
  latencyP95Seconds: number | null;
}

export interface ProvisioningSnapshot {
  windowHours: number | null;
  collectedAt: string | null; // cycle timestamp shared by every row
  fleet: ProvisionFleet | null;
  pools: ProvisionPool[];
}

export interface ProbesSnapshot {
  checks: ProbeCheck[];
  provisioning: ProvisioningSnapshot;
  fetchedAt: string;
  error?: string;
}

// Provisioning rows: friendly types, with collected_at as epoch-ms so Arrow
// hands us a plain number rather than a timestamp object, and the JSON label
// object kept as a string for per-row decode.
interface MetricRow {
  metric: string;
  value: number;
  labels: string; // JSON object string
  collected_ms: number;
}

// Health-check rows: the probe label is projected into its own column by
// checksSql, so there is no labels blob to decode here.
interface CheckRow {
  probe: string;
  metric: string;
  value: number;
  collected_ms: number;
}

function emptyProvisioning(): ProvisioningSnapshot {
  return { windowHours: null, collectedAt: null, fleet: null, pools: [] };
}

const checksSql = (cutoff: string) => `
  WITH recent AS (
    SELECT
      json_extract_string(labels, '$.probe') AS probe,
      metric,
      value,
      collected_at,
      ROW_NUMBER() OVER (
        PARTITION BY json_extract_string(labels, '$.probe'), metric
        ORDER BY collected_at DESC
      ) AS rn
    FROM "${METRICS_NAMESPACE}"
    WHERE metric IN ('${METRIC_UP}', '${METRIC_LATENCY_MS}')
      AND collected_at >= TIMESTAMP '${cutoff}'
  )
  SELECT probe, metric, value::DOUBLE AS value, epoch_ms(collected_at)::BIGINT AS collected_ms
  FROM recent
  WHERE rn = 1 AND probe IS NOT NULL
`;

const provisioningSql = (cutoff: string) => `
  SELECT metric, value::DOUBLE AS value, labels, epoch_ms(collected_at)::BIGINT AS collected_ms
  FROM "${METRICS_NAMESPACE}"
  WHERE metric LIKE '${PROVISION_PREFIX}%'
    AND collected_at >= TIMESTAMP '${cutoff}'
    AND collected_at = (
      SELECT MAX(collected_at) FROM "${METRICS_NAMESPACE}"
      WHERE metric LIKE '${PROVISION_PREFIX}%' AND collected_at >= TIMESTAMP '${cutoff}'
    )
`;

function asMetricRows(rows: Record<string, unknown>[]): MetricRow[] {
  return rows.map((r) => ({
    metric: String(r.metric),
    value: Number(r.value),
    labels: String(r.labels ?? "{}"),
    collected_ms: Number(r.collected_ms),
  }));
}

function asCheckRows(rows: Record<string, unknown>[]): CheckRow[] {
  return rows.map((r) => ({
    probe: String(r.probe),
    metric: String(r.metric),
    value: Number(r.value),
    collected_ms: Number(r.collected_ms),
  }));
}

// Latest probe_up (+ matching probe_latency_ms) per synthetic health check.
// Only probes that emit probe_up appear — the gauge collectors
// (provisioning/workers/jobs) emit probe_up solely when they fail, so a
// failing gauge surfaces here too, but a healthy one stays out of the list.
function parseChecks(rows: CheckRow[]): ProbeCheck[] {
  const up = new Map<string, { value: number; collectedMs: number }>();
  const latency = new Map<string, number>();
  for (const row of rows) {
    if (row.metric === METRIC_UP) up.set(row.probe, { value: row.value, collectedMs: row.collected_ms });
    else if (row.metric === METRIC_LATENCY_MS) latency.set(row.probe, row.value);
  }
  return [...up.entries()]
    .map(([probe, { value, collectedMs }]) => ({
      probe,
      up: value === 1,
      latencyMs: latency.has(probe) ? Math.round(latency.get(probe)!) : null,
      collectedAt: new Date(collectedMs).toISOString(),
    }))
    .sort((a, b) => a.probe.localeCompare(b.probe));
}

// Decode a row's JSON label blob. The probes write it via json.dumps, so a
// parse failure is a real anomaly (schema drift / truncation), not an
// expected case — log it rather than letting the row's scope/pool fields
// silently vanish from the rollup. The empty fallback keeps one bad row from
// sinking the whole snapshot.
function safeLabels(raw: string): Record<string, string> {
  try {
    return JSON.parse(raw) as Record<string, string>;
  } catch (err) {
    console.warn(`probes: unparseable labels ${JSON.stringify(raw.slice(0, 200))}: ${(err as Error).message}`);
    return {};
  }
}

function blankFleet(): ProvisionFleet {
  return {
    ready: 0,
    stockout: 0,
    error: 0,
    preempted: 0,
    outcomes: 0,
    successRatio: null,
    poolsPlacing: 0,
    poolsStockoutDead: 0,
    latencyP50Seconds: null,
    latencyP95Seconds: null,
  };
}

function blankPool(resourceType: string, scaleGroup: string, zone: string): ProvisionPool {
  return {
    resourceType,
    scaleGroup,
    zone,
    ready: 0,
    stockout: 0,
    error: 0,
    preempted: 0,
    outcomes: 0,
    successRatio: null,
    latencyP50Seconds: null,
    latencyP95Seconds: null,
  };
}

// Roll the latest provisioning cycle's rows into fleet + per-pool series.
// Every row in the cycle shares one collected_at; fleet rows carry
// scope=fleet, pool rows carry resource_type/scale_group/zone, and latency
// rows additionally carry a quantile label (p50/p95).
function parseProvisioning(rows: MetricRow[]): ProvisioningSnapshot {
  if (rows.length === 0) return emptyProvisioning();

  const fleet = blankFleet();
  const pools = new Map<string, ProvisionPool>();
  let windowHours: number | null = null;

  for (const row of rows) {
    const labels = safeLabels(row.labels);
    const quantile = labels.quantile;
    if (labels.scope === FLEET_SCOPE) {
      applyFleetMetric(fleet, row.metric, row.value, quantile);
      if (row.metric === METRIC_PROVISION_WINDOW_HOURS) windowHours = row.value;
      continue;
    }
    if (labels.resource_type && labels.scale_group && labels.zone) {
      const key = `${labels.resource_type} ${labels.scale_group} ${labels.zone}`;
      const pool = pools.get(key) ?? blankPool(labels.resource_type, labels.scale_group, labels.zone);
      applyCountMetric(pool, row.metric, row.value, quantile);
      pools.set(key, pool);
    }
  }

  for (const pool of pools.values()) {
    pool.successRatio = pool.outcomes > 0 ? pool.ready / pool.outcomes : null;
  }

  return {
    windowHours,
    collectedAt: new Date(rows[0]!.collected_ms).toISOString(),
    fleet,
    pools: [...pools.values()].sort(
      (a, b) =>
        a.resourceType.localeCompare(b.resourceType) ||
        a.scaleGroup.localeCompare(b.scaleGroup) ||
        a.zone.localeCompare(b.zone),
    ),
  };
}

// The outcome counts + latency quantiles shared by the fleet and per-pool
// series — both ProvisionFleet and ProvisionPool structurally satisfy this.
interface ProvisionCounts {
  ready: number;
  stockout: number;
  error: number;
  preempted: number;
  outcomes: number;
  latencyP50Seconds: number | null;
  latencyP95Seconds: number | null;
}

// Apply a count/latency metric common to both series; ignores fleet-only
// metrics (success ratio, pool tallies, window) which applyFleetMetric owns.
function applyCountMetric(target: ProvisionCounts, metric: string, value: number, quantile?: string): void {
  switch (metric) {
    case METRIC_PROVISION_READY:
      target.ready = value;
      break;
    case METRIC_PROVISION_STOCKOUT:
      target.stockout = value;
      break;
    case METRIC_PROVISION_ERROR:
      target.error = value;
      break;
    case METRIC_PROVISION_PREEMPTED:
      target.preempted = value;
      break;
    case METRIC_PROVISION_OUTCOMES:
      target.outcomes = value;
      break;
    case METRIC_PROVISION_LATENCY_SECONDS:
      if (quantile === "p50") target.latencyP50Seconds = value;
      else if (quantile === "p95") target.latencyP95Seconds = value;
      break;
  }
}

function applyFleetMetric(fleet: ProvisionFleet, metric: string, value: number, quantile?: string): void {
  applyCountMetric(fleet, metric, value, quantile);
  switch (metric) {
    case METRIC_PROVISION_SUCCESS_RATIO:
      fleet.successRatio = value;
      break;
    case METRIC_PROVISION_POOLS_PLACING:
      fleet.poolsPlacing = value;
      break;
    case METRIC_PROVISION_POOLS_STOCKOUT_DEAD:
      fleet.poolsStockoutDead = value;
      break;
  }
}

export async function probesSnapshot(): Promise<ProbesSnapshot> {
  const fetchedAt = new Date().toISOString();
  const now = Date.now();
  const checksCutoff = sqlTimestampUtc(new Date(now - CHECKS_LOOKBACK_MS));
  const provisioningCutoff = sqlTimestampUtc(new Date(now - PROVISIONING_LOOKBACK_MS));
  try {
    const [checkRows, provisioningRows] = await Promise.all([
      queryFinelog(checksSql(checksCutoff)),
      queryFinelog(provisioningSql(provisioningCutoff)),
    ]);
    return {
      checks: parseChecks(asCheckRows(checkRows)),
      provisioning: parseProvisioning(asMetricRows(provisioningRows)),
      fetchedAt,
    };
  } catch (err) {
    return {
      checks: [],
      provisioning: emptyProvisioning(),
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
