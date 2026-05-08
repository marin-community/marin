// Iris worker counts via ExecuteRawQuery against the controller's SQLite.
//
// Two round-trips per snapshot:
//   1. overall healthy-worker count + resource aggregates (CPU millicores,
//      memory bytes, TPU chips). Everything is gated on `healthy=1` only —
//      we intentionally do NOT filter on `active=1` because the panel
//      reports what's currently alive regardless of whether it's assigned
//      to work.
//   2. per-region breakdown, joined with worker_attributes on the "region"
//      key (verified present on the marin controller) so we don't have to
//      parse scale_group strings.
//
// CPU and memory are reported as "currently unallocated" (total - committed)
// because that's the useful number for scheduling headroom. TPU is
// reported as raw chip count across healthy workers; iris schedules TPU
// at whole-worker granularity so "committed vs free chips" collapses to
// "idle VMs × chips per VM" which is a small and usually misleading
// number on a busy cluster.
//
// History lives in a separate ring buffer (server/history.ts); this file
// only ever returns the current snapshot.

import { executeRawQuery } from "./controllerQuery.js";

const TOTALS_SQL = `
  SELECT
    SUM(CASE WHEN healthy=1 THEN 1 ELSE 0 END) AS healthy,
    COALESCE(SUM(CASE WHEN healthy=1 THEN total_cpu_millicores      ELSE 0 END), 0) AS cpu_total,
    COALESCE(SUM(CASE WHEN healthy=1 THEN committed_cpu_millicores  ELSE 0 END), 0) AS cpu_used,
    COALESCE(SUM(CASE WHEN healthy=1 THEN total_memory_bytes        ELSE 0 END), 0) AS mem_total,
    COALESCE(SUM(CASE WHEN healthy=1 THEN committed_mem_bytes       ELSE 0 END), 0) AS mem_used,
    COALESCE(SUM(CASE WHEN healthy=1 THEN total_tpu_count           ELSE 0 END), 0) AS chips_total
  FROM workers
`;

const REGION_SQL = `
  SELECT
    wa.str_value AS region,
    SUM(CASE WHEN w.healthy=1 THEN 1 ELSE 0 END) AS healthy
  FROM workers w
  JOIN worker_attributes wa
    ON wa.worker_id = w.worker_id AND wa.key = 'region'
  GROUP BY wa.str_value
  ORDER BY healthy DESC
`;

export interface WorkerResourceTotals {
  cpuAvailableMillicores: number;
  memoryAvailableBytes: number;
  // "chips" = total TPU chips across all healthy workers, NOT
  // "available" chips (which on a busy cluster collapses to a tiny
  // number because iris commits at whole-VM granularity).
  chipsTotal: number;
}

export interface WorkerRegionCount {
  region: string;
  healthy: number;
}

export interface WorkersSnapshot {
  healthy: number;
  resources: WorkerResourceTotals;
  byRegion: WorkerRegionCount[];
  fetchedAt: string;
  error?: string;
}

export interface WorkerSample {
  t: number; // epoch millis
  // Per-region healthy worker count at the sample time. Flat map keyed
  // by region name so the frontend can feed recharts directly (each
  // region becomes a <Line dataKey={region} />).
  regions: Record<string, number>;
}

function emptyResources(): WorkerResourceTotals {
  return {
    cpuAvailableMillicores: 0,
    memoryAvailableBytes: 0,
    chipsTotal: 0,
  };
}

// Hard ceiling so the TTLCache inflight promise can never hang indefinitely.
// If the inner queries don't settle in 20s (10s per-query timeout + margin),
// we return an error snapshot instead of blocking all future callers.
const SNAPSHOT_DEADLINE_MS = 20_000;

export async function workerSnapshot(): Promise<WorkersSnapshot> {
  const fetchedAt = new Date().toISOString();
  try {
    const [totalsResult, regionsResult] = await Promise.race([
      Promise.all([executeRawQuery(TOTALS_SQL), executeRawQuery(REGION_SQL)]),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("workerSnapshot deadline exceeded")), SNAPSHOT_DEADLINE_MS),
      ),
    ]);
    const totalsRow = totalsResult.rows[0];
    if (!totalsRow) {
      return {
        healthy: 0,
        resources: emptyResources(),
        byRegion: [],
        fetchedAt,
        error: "empty result from workers totals query",
      };
    }
    const [
      healthyRaw,
      cpuTotalRaw,
      cpuUsedRaw,
      memTotalRaw,
      memUsedRaw,
      chipsTotalRaw,
    ] = totalsRow;
    const cpuTotal = Number(cpuTotalRaw ?? 0);
    const cpuUsed = Number(cpuUsedRaw ?? 0);
    const memTotal = Number(memTotalRaw ?? 0);
    const memUsed = Number(memUsedRaw ?? 0);
    const chipsTotal = Number(chipsTotalRaw ?? 0);

    const byRegion: WorkerRegionCount[] = regionsResult.rows.map((row) => {
      const [regionRaw, healthyRegionRaw] = row;
      return {
        region: String(regionRaw ?? "unknown"),
        healthy: Number(healthyRegionRaw ?? 0),
      };
    });

    return {
      healthy: Number(healthyRaw ?? 0),
      resources: {
        cpuAvailableMillicores: Math.max(0, cpuTotal - cpuUsed),
        memoryAvailableBytes: Math.max(0, memTotal - memUsed),
        chipsTotal,
      },
      byRegion,
      fetchedAt,
    };
  } catch (err) {
    return {
      healthy: 0,
      resources: emptyResources(),
      byRegion: [],
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
