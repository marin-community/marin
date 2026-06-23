// Finelog-backed time series for the Workers + provisioning history charts.
//
// The infra/probes canary writes one flat row per metric per cycle to the
// `infra.canary.metrics` namespace (see infra/probes/src/cluster.py and
// provisioning.py): `worker_healthy` per region (and a fleet total) every 60s,
// and the `provision_*` fleet/per-pool gauges every 15min. This module rolls
// the trailing 24h of those rows into the two chart series the frontend reads.
//
// It replaces the in-process worker-count ring buffer, whose history was lost
// on every Cloud Run restart (see README "Known limitations"). The canary now
// owns this history durably in finelog, so the dashboard just queries it.

import { decodeLabels, microsToMillis, queryFinelog, sqlTimestampUtc } from "./finelogQuery.js";

const METRICS_NAMESPACE = "infra.canary.metrics";
const HISTORY_WINDOW_MS = 24 * 60 * 60 * 1000;

const METRIC_WORKER_HEALTHY = "worker_healthy";
const METRIC_PROVISION_READY = "provision_ready";
const METRIC_PROVISION_OUTCOMES = "provision_outcomes";
const FLEET_SCOPE = "fleet";

// Per-region healthy worker count at one sample time. Flat map keyed by region
// so the frontend feeds recharts directly (each region → a <Line dataKey>).
export interface WorkerSample {
  t: number; // epoch millis
  regions: Record<string, number>;
}

export interface WorkersHistoryResponse {
  samples: WorkerSample[];
  windowMs: number;
  fetchedAt: string;
  error?: string;
}

// One provisioning cycle's create-success ratio (ready / resolved attempts).
// `fleet` is the cluster-wide average (null when the cycle resolved zero
// attempts); `regions` carries the per-region ratio, keyed by region name and
// omitting any region with zero attempts that cycle. Ratios are 0..1.
export interface ProvisioningHistorySample {
  t: number; // epoch millis
  fleet: number | null;
  regions: Record<string, number>;
}

export interface ProvisioningHistoryResponse {
  samples: ProvisioningHistorySample[];
  windowMs: number;
  fetchedAt: string;
  error?: string;
}

interface MetricRow {
  metric: string;
  value: number;
  labels: string;
  collected_us: number;
}

function asRows(rows: Record<string, unknown>[]): MetricRow[] {
  return rows.map((r) => ({
    metric: String(r.metric),
    value: Number(r.value),
    labels: String(r.labels ?? "{}"),
    collected_us: Number(r.collected_us),
  }));
}

// SQL is Apache DataFusion (finelog's read engine), NOT DuckDB: no JSON
// functions (labels are decoded in JS), and collected_at is read out as epoch
// micros via arrow_cast(...,'Int64') so Arrow hands back a plain integer.
const workersSql = (cutoff: string) => `
  SELECT labels, value, arrow_cast(collected_at, 'Int64') AS collected_us
  FROM "${METRICS_NAMESPACE}"
  WHERE metric = '${METRIC_WORKER_HEALTHY}'
    AND collected_at >= TIMESTAMP '${cutoff}'
  ORDER BY collected_at
`;

// Pull both the fleet-scoped and per-pool ready/outcomes counts; the success
// ratio is computed here (ready / outcomes) for the fleet average and rolled
// up per region from the per-pool rows, rather than reading the probe's
// pre-baked fleet-only provision_success_ratio.
const provisioningSql = (cutoff: string) => `
  SELECT metric, value, labels, arrow_cast(collected_at, 'Int64') AS collected_us
  FROM "${METRICS_NAMESPACE}"
  WHERE metric IN ('${METRIC_PROVISION_READY}', '${METRIC_PROVISION_OUTCOMES}')
    AND collected_at >= TIMESTAMP '${cutoff}'
  ORDER BY collected_at
`;

// Group the per-region worker_healthy rows by cycle timestamp. The fleet-total
// rows (scope=fleet) are dropped — the chart plots per-region lines, matching
// what the in-process buffer used to feed it.
function parseWorkerSamples(rows: MetricRow[]): WorkerSample[] {
  const byTime = new Map<number, Record<string, number>>();
  for (const row of rows) {
    const region = decodeLabels(row.labels).region;
    if (!region) continue;
    const t = microsToMillis(row.collected_us);
    const regions = byTime.get(t) ?? {};
    regions[region] = row.value;
    byTime.set(t, regions);
  }
  return [...byTime.entries()].sort((a, b) => a[0] - b[0]).map(([t, regions]) => ({ t, regions }));
}

// GCP zone → region: drop the trailing "-<letter>" (us-east5-a → us-east5),
// matching the region names the worker_healthy gauge already uses.
function regionOfZone(zone: string): string {
  return zone.replace(/-[a-z]$/, "");
}

interface Counts {
  ready: number;
  outcomes: number;
}

function ratio({ ready, outcomes }: Counts): number | null {
  return outcomes > 0 ? ready / outcomes : null;
}

// Group the per-pool ready/outcomes counts by cycle timestamp into a fleet
// average plus a per-region rollup. provision_outcomes already excludes
// preemptions (it counts resolved create attempts), so ready / outcomes is the
// create-success ratio. The fleet rows (scope=fleet) carry the cluster total;
// per-pool rows are summed across pools sharing a region.
function parseProvisioningSamples(rows: MetricRow[]): ProvisioningHistorySample[] {
  const fleetByTime = new Map<number, Counts>();
  const regionByTime = new Map<number, Map<string, Counts>>();

  for (const row of rows) {
    const labels = decodeLabels(row.labels);
    const t = microsToMillis(row.collected_us);
    if (labels.scope === FLEET_SCOPE) {
      const counts = fleetByTime.get(t) ?? { ready: 0, outcomes: 0 };
      addCount(counts, row.metric, row.value);
      fleetByTime.set(t, counts);
    } else if (labels.zone) {
      const region = regionOfZone(labels.zone);
      const regions = regionByTime.get(t) ?? new Map<string, Counts>();
      const counts = regions.get(region) ?? { ready: 0, outcomes: 0 };
      addCount(counts, row.metric, row.value);
      regions.set(region, counts);
      regionByTime.set(t, regions);
    }
  }

  const times = [...new Set([...fleetByTime.keys(), ...regionByTime.keys()])].sort((a, b) => a - b);
  return times.map((t) => {
    const regions: Record<string, number> = {};
    for (const [region, counts] of regionByTime.get(t) ?? []) {
      const r = ratio(counts);
      if (r !== null) regions[region] = r;
    }
    const fleetCounts = fleetByTime.get(t);
    return { t, fleet: fleetCounts ? ratio(fleetCounts) : null, regions };
  });
}

function addCount(counts: Counts, metric: string, value: number): void {
  if (metric === METRIC_PROVISION_READY) counts.ready += value;
  else if (metric === METRIC_PROVISION_OUTCOMES) counts.outcomes += value;
}

export async function workersHistory(): Promise<WorkersHistoryResponse> {
  const fetchedAt = new Date().toISOString();
  const cutoff = sqlTimestampUtc(new Date(Date.now() - HISTORY_WINDOW_MS));
  try {
    const rows = await queryFinelog(workersSql(cutoff));
    return { samples: parseWorkerSamples(asRows(rows)), windowMs: HISTORY_WINDOW_MS, fetchedAt };
  } catch (err) {
    return { samples: [], windowMs: HISTORY_WINDOW_MS, fetchedAt, error: (err as Error).message };
  }
}

export async function provisioningHistory(): Promise<ProvisioningHistoryResponse> {
  const fetchedAt = new Date().toISOString();
  const cutoff = sqlTimestampUtc(new Date(Date.now() - HISTORY_WINDOW_MS));
  try {
    const rows = await queryFinelog(provisioningSql(cutoff));
    return { samples: parseProvisioningSamples(asRows(rows)), windowMs: HISTORY_WINDOW_MS, fetchedAt };
  } catch (err) {
    return { samples: [], windowMs: HISTORY_WINDOW_MS, fetchedAt, error: (err as Error).message };
  }
}
