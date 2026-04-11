// Iris worker counts via ExecuteRawQuery against the controller's SQLite.
//
// One round-trip returns the aggregate counts we display on the dashboard.
// The `workers.healthy` and `workers.active` columns are both booleans
// stored as INTEGER; a worker is "available" iff both are 1, matching the
// iris Fleet tab semantic (lib/iris/src/iris/cluster/web/FleetTab.vue).
//
// History lives in a separate ring buffer (server/history.ts); this file
// only ever returns the current snapshot. Set WORKERS_FIXTURE=1 (or the
// shared IRIS_FIXTURE=1) to short-circuit with synthetic data for UI
// work without a tunnel to the controller.

import { executeRawQuery } from "./controllerQuery.js";

const FIXTURE_MODE = process.env.WORKERS_FIXTURE === "1" || process.env.IRIS_FIXTURE === "1";

const COUNT_SQL = `
  SELECT
    SUM(CASE WHEN healthy=1 AND active=1 THEN 1 ELSE 0 END) AS available,
    COUNT(*) AS total
  FROM workers
`;

export interface WorkersSnapshot {
  available: number;
  total: number;
  fetchedAt: string;
  error?: string;
}

export interface WorkerSample {
  t: number; // epoch millis
  available: number;
  total: number;
}

function fixtureSnapshot(): WorkersSnapshot {
  const total = 720;
  const available = total - Math.floor(Math.random() * 30);
  return {
    available,
    total,
    fetchedAt: new Date().toISOString(),
  };
}

// Prefill for fixture mode: 24h of synthetic samples at 30s cadence so
// the chart renders immediately during local UI dev.
export function fixtureHistory(capacity: number, intervalMs: number): WorkerSample[] {
  const now = Date.now();
  const samples: WorkerSample[] = [];
  for (let i = 0; i < capacity; i++) {
    const t = now - (capacity - 1 - i) * intervalMs;
    // Slow sinusoid so the chart looks alive without being misleading.
    const base = 720;
    const wobble = Math.round(Math.sin((i / capacity) * Math.PI * 4) * 20);
    const jitter = Math.floor(Math.random() * 6);
    const total = base + wobble;
    const available = Math.max(0, total - jitter);
    samples.push({ t, available, total });
  }
  return samples;
}

export function fixtureEnabled(): boolean {
  return FIXTURE_MODE;
}

export async function workerSnapshot(): Promise<WorkersSnapshot> {
  const fetchedAt = new Date().toISOString();
  if (FIXTURE_MODE) {
    return fixtureSnapshot();
  }
  try {
    const { rows } = await executeRawQuery(COUNT_SQL);
    const row = rows[0];
    if (!row) {
      return { available: 0, total: 0, fetchedAt, error: "empty result from workers query" };
    }
    const [availableRaw, totalRaw] = row;
    return {
      available: Number(availableRaw ?? 0),
      total: Number(totalRaw ?? 0),
      fetchedAt,
    };
  } catch (err) {
    return {
      available: 0,
      total: 0,
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
