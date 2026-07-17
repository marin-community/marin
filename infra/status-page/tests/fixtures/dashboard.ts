import type {
  BuildsResponse,
  IrisStatus,
  JobsSnapshot,
  ProbesSnapshot,
  ProvisioningHistoryResponse,
  ServiceHealthResponse,
  WandbSnapshot,
  WorkersHistoryResponse,
  WorkersSnapshot,
} from "../../web/src/api";
import {
  NIGHTLY_LANES,
  type UtcWeekday,
} from "../../server/sources/nightlyConfig.js";
import {
  projectNightlies,
  type NightlyLaneSnapshot,
  type NightlyRun,
} from "../../server/sources/nightlyProjection.js";

const NOW = "2026-07-17T12:30:00.000Z";
const FIXED_NOW = new Date(NOW);
const SECONDS_PER_MINUTE = 60;
const HISTORICAL_DATES = [
  "2026-07-16",
  "2026-07-15",
  "2026-07-14",
  "2026-07-13",
  "2026-07-12",
  "2026-07-11",
] as const;

const snapshots = new Map<string, NightlyLaneSnapshot>(
  NIGHTLY_LANES.map((lane) => [
    lane.id,
    { laneId: lane.id, fetchedAt: NOW, runs: [] },
  ]),
);
let nextRunId = 100;

function addRun(
  laneId: string,
  date: string,
  durationSeconds: number,
  options: { status?: string; conclusion?: string | null; recovered?: boolean } = {},
): void {
  const lane = NIGHTLY_LANES.find((candidate) => candidate.id === laneId);
  const snapshot = snapshots.get(laneId);
  if (!lane || !snapshot) throw new Error(`Unknown fixture lane: ${laneId}`);

  const id = nextRunId;
  nextRunId += 1;
  const status = options.status ?? "completed";
  const end = status === "completed" ? new Date(`${date}T12:00:00.000Z`) : FIXED_NOW;
  const started = new Date(end.getTime() - durationSeconds * 1000);
  const created = new Date(started.getTime() - SECONDS_PER_MINUTE * 1000);
  const url = `https://github.com/${lane.repository}/actions/runs/${id}`;
  const run: NightlyRun = {
    id,
    status,
    conclusion: options.conclusion === undefined ? "success" : options.conclusion,
    sha: `${id}`.padStart(40, "a"),
    createdAt: created.toISOString(),
    runStartedAt: started.toISOString(),
    updatedAt: end.toISOString(),
    url,
    runAttempt: options.recovered ? 2 : 1,
    event: "schedule",
    headBranch: lane.branch,
    actor: "github-actions",
  };
  snapshot.runs.push(run);

  if (options.recovered) {
    snapshot.attemptsByRunId = {
      ...snapshot.attemptsByRunId,
      [String(id)]: [
        {
          attempt: 1,
          status: "completed",
          conclusion: "failure",
          runStartedAt: started.toISOString(),
          updatedAt: new Date(started.getTime() + 5 * SECONDS_PER_MINUTE * 1000).toISOString(),
          url,
        },
      ],
    };
  }
}

for (const lane of NIGHTLY_LANES) {
  for (const date of HISTORICAL_DATES) {
    const weekday = new Date(`${date}T00:00:00.000Z`).getUTCDay();
    if (!lane.schedule.weekdays.includes(weekday as UtcWeekday)) continue;
    if (lane.activeFrom && date < lane.activeFrom) continue;
    if (lane.activeUntil && date > lane.activeUntil) continue;
    addRun(lane.id, date, lane.expectedDuration?.minSeconds ?? 10 * SECONDS_PER_MINUTE);
  }
}

addRun("tpu-ferry", "2026-07-17", 125 * SECONDS_PER_MINUTE);
addRun("cw-gpu-ferry", "2026-07-17", 22 * SECONDS_PER_MINUTE, { conclusion: "failure" });
addRun("grug-multislice", "2026-07-17", 11 * SECONDS_PER_MINUTE);
addRun("datakit-t1", "2026-07-17", 92 * SECONDS_PER_MINUTE);
addRun("datakit-t2", "2026-07-17", 135 * SECONDS_PER_MINUTE);
addRun("evalchemy", "2026-07-17", 23 * SECONDS_PER_MINUTE, {
  status: "in_progress",
  conclusion: null,
});
addRun("harbor", "2026-07-17", 8 * SECONDS_PER_MINUTE, { recovered: true });
addRun("marinskyrl", "2026-07-17", 51);
addRun("vllm-gpu", "2026-07-17", 71);

export const NIGHTLIES_FIXTURE = projectNightlies(
  NIGHTLY_LANES,
  [...snapshots.values()],
  FIXED_NOW,
);
const BUILDS_FIXTURE: BuildsResponse = {
  commits: [{ oid: "abcdef1234567890", shortOid: "abcdef1", headline: "Keep the dashboard calm and useful", committedAt: "2026-07-17T12:00:00.000Z", author: "marin", authorAvatarUrl: null, url: "https://github.com/marin-community/marin/commit/abcdef1234567890", state: "SUCCESS" }],
  successRate: 1,
  fetchedAt: NOW,
};

const IRIS_FIXTURE: IrisStatus = {
  cluster: "marin",
  reachable: true,
  latencyMs: 8,
  pingPercentiles: { p50: 7, p90: 10, p99: 13 },
  pingSampleCount: 120,
  pingSpanMs: 3_600_000,
  pingWindowMs: 3_600_000,
  controllerUrl: "http://iris.internal",
  fetchedAt: NOW,
};

const WORKERS_FIXTURE: WorkersSnapshot = {
  healthy: 42,
  resources: { cpuTotalMillicores: 960_000, memoryTotalBytes: 8_796_093_022_208, chipsTotal: 256 },
  byRegion: [{ region: "us-central1", healthy: 42 }],
  fetchedAt: NOW,
};

const WORKERS_HISTORY_FIXTURE: WorkersHistoryResponse = { samples: [], windowMs: 86_400_000, fetchedAt: NOW };
const PROVISIONING_HISTORY_FIXTURE: ProvisioningHistoryResponse = { samples: [], windowMs: 86_400_000, fetchedAt: NOW };
const CONTROL_PLANE_FIXTURE: ServiceHealthResponse = { environment: "prod", series: [], latest: [], samples: [], summarySamples: [], aggregationWindowMs: 300_000, summaryPointIntervalMs: 30_000, windowMs: 86_400_000, fetchedAt: NOW };
const JOBS_FIXTURE: JobsSnapshot = { inflight: { total: 2, byState: [{ state: 2, name: "running", count: 2 }] }, last24h: { total: 24, byState: [{ state: 4, name: "succeeded", count: 23 }, { state: 5, name: "failed", count: 1 }] }, windowMs: 86_400_000, fetchedAt: NOW };
const PROBES_FIXTURE: ProbesSnapshot = { checks: [{ probe: "controller-ping", up: true, latencyMs: 8, collectedAt: NOW }], provisioning: { windowHours: 3, collectedAt: NOW, fleet: { ready: 8, stockout: 0, error: 0, preempted: 1, outcomes: 8, successRatio: 1, poolsPlacing: 2, poolsStockoutDead: 0, latencyP50Seconds: 180, latencyP95Seconds: 300 }, pools: [] }, fetchedAt: NOW };
const WANDB_FIXTURE: WandbSnapshot = { reportTitle: "67B-A2B MoE on 10T tokens", reportUrl: "https://wandb.ai/marin-community/marin_moe", charts: [], fetchedAt: NOW };

export const API_FIXTURES: Record<string, unknown> = {
  "/api/nightlies": NIGHTLIES_FIXTURE,
  "/api/builds": BUILDS_FIXTURE,
  "/api/iris": IRIS_FIXTURE,
  "/api/control-plane/health": CONTROL_PLANE_FIXTURE,
  "/api/workers": WORKERS_FIXTURE,
  "/api/workers/history": WORKERS_HISTORY_FIXTURE,
  "/api/provisioning/history": PROVISIONING_HISTORY_FIXTURE,
  "/api/jobs": JOBS_FIXTURE,
  "/api/probes": PROBES_FIXTURE,
  "/api/wandb": WANDB_FIXTURE,
};
