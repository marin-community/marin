// Iris job counts split into two buckets via ExecuteRawQuery:
//
// - inflight: every root job currently in PENDING / BUILDING / RUNNING,
//   regardless of when it was submitted. Long-running experiments that
//   started >24h ago still show up here, which is what the iris Fleet
//   dashboard displays.
// - last24h: terminal root jobs (succeeded / failed / killed /
//   worker_failed / unschedulable) that finished in the last 24h, via
//   finished_at_ms.
//
// The controller's `jobs` table (see lib/iris/src/iris/cluster/controller/schema.py)
// stores state as an integer matching the JobState enum in
// lib/iris/src/iris/rpc/job.proto:182. We group by integer in SQL (so
// the query plan stays simple) and translate to friendly names on the way
// out. If the enum grows, unknown values fall through as "state_N".

import { executeRawQuery } from "./controllerQuery.js";

const WINDOW_MS = 24 * 60 * 60 * 1000;

// Filters to root jobs only (parent_job_id IS NULL) so the count
// reflects what users explicitly submitted rather than the full
// fan-out of every sub-job each root spawns. A single experiment can
// easily create hundreds of child jobs, which would otherwise make
// the panel a misleading measure of "team activity". Confirmed
// against the live marin controller: parent_job_id is NULL for root
// jobs and non-null for children (no empty-string variant).
//
// In-flight states (PENDING=1, BUILDING=2, RUNNING=3) are always
// counted regardless of when the job was submitted — long-running
// experiments that started >24h ago should still show up in the
// "running" bar, which matches what the iris Fleet dashboard shows.
// Terminal states (SUCCEEDED=4, FAILED=5, KILLED=6, WORKER_FAILED=7,
// UNSCHEDULABLE=8) are filtered to jobs that finished in the last
// 24h via finished_at_ms, which is always populated for terminal
// jobs (verified on the marin controller).
const BREAKDOWN_SQL = `
  SELECT state, COUNT(*) AS n
  FROM jobs
  WHERE parent_job_id IS NULL
    AND (
      state IN (1, 2, 3)
      OR (
        state IN (4, 5, 6, 7, 8)
        AND finished_at_ms > (strftime('%s', 'now') * 1000 - ${WINDOW_MS})
      )
    )
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

// States <= 3 are in-flight (pending / building / running); the rest
// are terminal (succeeded / failed / killed / worker_failed /
// unschedulable). The panel renders them as two separate sections
// because their semantics differ — in-flight is a live snapshot,
// terminal is a 24h window.
const IN_FLIGHT_STATES: ReadonlySet<number> = new Set([1, 2, 3]);

export interface JobStateCount {
  state: number;
  name: string;
  count: number;
}

export interface JobBucket {
  total: number;
  byState: JobStateCount[];
}

export interface JobsSnapshot {
  inflight: JobBucket;
  last24h: JobBucket;
  windowMs: number;
  fetchedAt: string;
  error?: string;
}

function stateName(state: number): string {
  return JOB_STATE_NAMES[state] ?? `state_${state}`;
}

function emptyBucket(): JobBucket {
  return { total: 0, byState: [] };
}

function bucketize(rows: JobStateCount[]): { inflight: JobBucket; last24h: JobBucket } {
  const inflight: JobStateCount[] = [];
  const last24h: JobStateCount[] = [];
  for (const row of rows) {
    (IN_FLIGHT_STATES.has(row.state) ? inflight : last24h).push(row);
  }
  const sortDesc = (a: JobStateCount, b: JobStateCount) => b.count - a.count;
  inflight.sort(sortDesc);
  last24h.sort(sortDesc);
  return {
    inflight: {
      total: inflight.reduce((sum, r) => sum + r.count, 0),
      byState: inflight,
    },
    last24h: {
      total: last24h.reduce((sum, r) => sum + r.count, 0),
      byState: last24h,
    },
  };
}

export async function jobsSnapshot(): Promise<JobsSnapshot> {
  const fetchedAt = new Date().toISOString();
  try {
    const { rows } = await executeRawQuery(BREAKDOWN_SQL);
    const rowsOut: JobStateCount[] = rows.map((row) => {
      const [stateRaw, countRaw] = row;
      const state = Number(stateRaw);
      return {
        state,
        name: stateName(state),
        count: Number(countRaw ?? 0),
      };
    });
    const buckets = bucketize(rowsOut);
    return { ...buckets, windowMs: WINDOW_MS, fetchedAt };
  } catch (err) {
    return {
      inflight: emptyBucket(),
      last24h: emptyBucket(),
      windowMs: WINDOW_MS,
      fetchedAt,
      error: (err as Error).message,
    };
  }
}
