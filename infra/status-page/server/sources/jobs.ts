// Iris job counts for the last 24h, grouped by state, via ExecuteRawQuery.
//
// The controller's `jobs` table (see lib/iris/src/iris/cluster/controller/schema.py)
// stores state as an integer matching the JobState enum in
// lib/iris/src/iris/rpc/job.proto:182. We group by integer in SQL (so
// the query plan stays simple) and translate to friendly names on the way
// out. If the enum grows, unknown values fall through as "state_N".

import { executeRawQuery } from "./controllerQuery.js";

const WINDOW_MS = 24 * 60 * 60 * 1000;

const BREAKDOWN_SQL = `
  SELECT state, COUNT(*) AS n
  FROM jobs
  WHERE submitted_at_ms > (strftime('%s', 'now') * 1000 - ${WINDOW_MS})
  GROUP BY state
`;

// Mirrors lib/iris/src/iris/rpc/job.proto:182. Keep in sync if the enum
// changes — the status page will surface unknown values as "state_N"
// rather than silently dropping them.
const JOB_STATE_NAMES: Record<number, string> = {
  0: "unspecified",
  1: "pending",
  2: "building",
  3: "running",
  4: "succeeded",
  5: "failed",
  6: "killed",
  7: "worker_failed",
  8: "unschedulable",
};

export interface JobStateCount {
  state: number;
  name: string;
  count: number;
}

export interface JobsSnapshot {
  total: number;
  windowMs: number;
  byState: JobStateCount[];
  fetchedAt: string;
  error?: string;
}

function stateName(state: number): string {
  return JOB_STATE_NAMES[state] ?? `state_${state}`;
}

export async function jobsSnapshot(): Promise<JobsSnapshot> {
  const fetchedAt = new Date().toISOString();
  try {
    const { rows } = await executeRawQuery(BREAKDOWN_SQL);
    const byState: JobStateCount[] = rows
      .map((row) => {
        const [stateRaw, countRaw] = row;
        const state = Number(stateRaw);
        return {
          state,
          name: stateName(state),
          count: Number(countRaw ?? 0),
        };
      })
      .sort((a, b) => b.count - a.count);
    const total = byState.reduce((sum, r) => sum + r.count, 0);
    return { total, windowMs: WINDOW_MS, byState, fetchedAt };
  } catch (err) {
    return {
      total: 0,
      windowMs: WINDOW_MS,
      byState: [],
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
