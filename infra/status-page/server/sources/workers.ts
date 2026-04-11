// Iris worker counts via ExecuteRawQuery against the controller's SQLite.
//
// One round-trip returns the aggregate counts we display on the dashboard.
// The `workers.healthy` and `workers.active` columns are both booleans
// stored as INTEGER; a worker is "available" iff both are 1, matching the
// iris Fleet tab semantic (lib/iris/src/iris/cluster/web/FleetTab.vue).
//
// History lives in a separate ring buffer (server/history.ts); this file
// only ever returns the current snapshot.

import { executeRawQuery } from "./controllerQuery.js";

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

export async function workerSnapshot(): Promise<WorkersSnapshot> {
  const fetchedAt = new Date().toISOString();
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
