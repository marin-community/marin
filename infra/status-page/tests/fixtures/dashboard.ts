import type {
  BuildsResponse,
  IrisStatus,
  JobsSnapshot,
  NightlyCell,
  NightlyDurationState,
  NightlyLane,
  NightlyResponse,
  ProbesSnapshot,
  ProvisioningHistoryResponse,
  ServiceHealthResponse,
  WandbSnapshot,
  WorkersHistoryResponse,
  WorkersSnapshot,
} from "../../web/src/api";

const NOW = "2026-07-17T12:30:00.000Z";
const DAILY_DATES = [
  ["2026-07-17", "Fri, Jul 17"],
  ["2026-07-16", "Thu, Jul 16"],
  ["2026-07-15", "Wed, Jul 15"],
  ["2026-07-14", "Tue, Jul 14"],
  ["2026-07-13", "Mon, Jul 13"],
  ["2026-07-12", "Sun, Jul 12"],
  ["2026-07-11", "Sat, Jul 11"],
] as const;

function lane(
  id: string,
  label: string,
  shortLabel: string,
  group: NightlyLane["group"],
  subgroup: NightlyLane["subgroup"],
  repository: string,
  workflowFile: string,
  scheduleLabel: string,
  expectedDuration?: { minSeconds: number; maxSeconds: number; provenance: string },
): NightlyLane {
  return {
    id,
    label,
    shortLabel,
    group,
    subgroup,
    repository,
    workflowFile,
    workflowUrl: `https://github.com/${repository}/blob/main/.github/workflows/${workflowFile}`,
    scheduleLabel,
    overdueGraceMinutes: scheduleLabel.startsWith("Mon") ? 480 : 240,
    overdueGraceProvenance: "Deterministic visual fixture",
    ...(expectedDuration ? { expectedDuration } : {}),
  };
}

const MINUTES = 60;
export const NIGHTLY_LANES_FIXTURE: NightlyLane[] = [
  lane("tpu-ferry", "TPU ferry", "TPU ferry", "marin", "training", "marin-community/marin", "marin-canary-ferry.yaml", "Daily 06:00 UTC", { minSeconds: 60 * MINUTES, maxSeconds: 195 * MINUTES, provenance: "fixture" }),
  lane("cw-gpu-ferry", "CoreWeave GPU ferry", "CW ferry", "marin", "training", "marin-community/marin", "marin-canary-ferry-coreweave.yaml", "Daily 10:00 UTC", { minSeconds: 15 * MINUTES, maxSeconds: 40 * MINUTES, provenance: "fixture" }),
  lane("grug-multislice", "Grug multislice", "Grug", "marin", "training", "marin-community/marin", "marin-canary-grug-multislice.yaml", "Daily 10:30 UTC"),
  lane("datakit-t1", "Datakit tier 1", "Data T1", "marin", "data", "marin-community/marin", "marin-canary-datakit-tier1.yaml", "Daily 06:30 UTC", { minSeconds: 65 * MINUTES, maxSeconds: 85 * MINUTES, provenance: "fixture" }),
  lane("datakit-t2", "Datakit tier 2", "Data T2", "marin", "data", "marin-community/marin", "marin-canary-datakit-tier2.yaml", "Daily 07:00 UTC", { minSeconds: 65 * MINUTES, maxSeconds: 85 * MINUTES, provenance: "fixture" }),
  lane("datakit-t3", "Datakit tier 3, Mondays", "Data T3 · Mon", "marin", "data", "marin-community/marin", "marin-canary-datakit-tier3.yaml", "Mon 01:00 UTC", { minSeconds: 70 * MINUTES, maxSeconds: 180 * MINUTES, provenance: "fixture" }),
  lane("cluster-smoke", "Cluster smoke", "Cluster", "marin", "cluster", "marin-community/marin", "marin-cluster-smoke.yaml", "Daily 07:30 UTC"),
  lane("evalchemy", "Evalchemy", "Evalchemy", "forks", "evaluation", "marin-community/evalchemy", "e2e-nightly.yaml", "Daily 07:00 UTC", { minSeconds: 14 * MINUTES, maxSeconds: 20 * MINUTES, provenance: "fixture" }),
  lane("harbor", "Harbor", "Harbor", "forks", "evaluation", "marin-community/harbor", "marin-nightly.yaml", "Daily 08:00 UTC", { minSeconds: 6 * MINUTES, maxSeconds: 12 * MINUTES, provenance: "fixture" }),
  lane("marinskyrl", "MarinSkyRL", "SkyRL", "forks", "rl", "marin-community/MarinSkyRL", "marin-nightly.yaml", "Daily 09:00 UTC"),
  lane("vllm-gpu", "vLLM GPU", "vLLM GPU", "forks", "inference", "marin-community/vllm", "marin-nightly.yaml", "Daily 10:00 UTC", { minSeconds: 6 * MINUTES, maxSeconds: 15 * MINUTES, provenance: "fixture" }),
  lane("tpu-inference", "TPU inference", "TPU infer", "forks", "inference", "marin-community/tpu-inference", "marin-e2e-nightly.yaml", "Daily 11:00 UTC", { minSeconds: 5 * MINUTES, maxSeconds: 10 * MINUTES, provenance: "fixture" }),
];

function emptyCell(
  laneId: string,
  date: string,
  state: Exclude<NightlyCell["state"], "run">,
): NightlyCell {
  return {
    laneId,
    date,
    expectedAt: state === "not-scheduled" ? null : `${date}T10:00:00.000Z`,
    state,
    due: state === "missing" || state === "unavailable",
    healthy: false,
    durationState: "not-applicable",
    sourceFetchedAt: NOW,
    ...(state === "unavailable" ? { sourceError: "Fixture GitHub outage" } : {}),
  };
}

function runCell(
  laneId: string,
  date: string,
  id: number,
  durationSeconds: number,
  durationState: NightlyDurationState,
  conclusion: string | null = "success",
  options: { status?: string; recovered?: boolean } = {},
): NightlyCell {
  const status = options.status ?? "completed";
  return {
    laneId,
    date,
    expectedAt: `${date}T10:00:00.000Z`,
    state: "run",
    due: true,
    healthy: status === "completed" && conclusion === "success" && durationState !== "too-short",
    durationState,
    sourceFetchedAt: NOW,
    run: {
      id,
      status,
      conclusion,
      sha: `${id}`.padStart(40, "a"),
      createdAt: `${date}T11:00:00.000Z`,
      runStartedAt: `${date}T11:00:00.000Z`,
      updatedAt: `${date}T11:30:00.000Z`,
      url: `https://github.com/marin-community/marin/actions/runs/${id}`,
      runAttempt: options.recovered ? 2 : 1,
      event: "schedule",
      headBranch: "main",
      actor: "github-actions",
      shaShort: `${id}`.padStart(7, "a"),
      durationSeconds,
      recovered: options.recovered ?? false,
      priorAttempts: options.recovered
        ? [{ attempt: 1, status: "completed", conclusion: "failure", runStartedAt: `${date}T10:30:00.000Z`, updatedAt: `${date}T10:35:00.000Z`, url: `https://github.com/marin-community/marin/actions/runs/${id}` }]
        : [],
    },
  };
}

const TODAY_CELLS: NightlyCell[] = [
  runCell("tpu-ferry", "2026-07-17", 101, 125 * MINUTES, "normal"),
  runCell("cw-gpu-ferry", "2026-07-17", 102, 22 * MINUTES, "normal", "failure"),
  runCell("grug-multislice", "2026-07-17", 103, 11 * MINUTES, "baseline-pending"),
  runCell("datakit-t1", "2026-07-17", 104, 92 * MINUTES, "slow"),
  runCell("datakit-t2", "2026-07-17", 105, 135 * MINUTES, "very-slow"),
  emptyCell("datakit-t3", "2026-07-17", "not-scheduled"),
  emptyCell("cluster-smoke", "2026-07-17", "missing"),
  runCell("evalchemy", "2026-07-17", 108, 23 * MINUTES, "slow", null, { status: "in_progress" }),
  runCell("harbor", "2026-07-17", 109, 8 * MINUTES, "normal", "success", { recovered: true }),
  runCell("marinskyrl", "2026-07-17", 110, 51, "baseline-pending"),
  runCell("vllm-gpu", "2026-07-17", 111, 71, "too-short"),
  emptyCell("tpu-inference", "2026-07-17", "not-yet-due"),
];

function historicalCells(date: string, rowIndex: number): NightlyCell[] {
  return NIGHTLY_LANES_FIXTURE.map((nightlyLane, laneIndex) => {
    if (nightlyLane.id === "datakit-t3" && date !== "2026-07-13") {
      return emptyCell(nightlyLane.id, date, "not-scheduled");
    }
    if (nightlyLane.id === "cluster-smoke") {
      return emptyCell(nightlyLane.id, date, "not-introduced");
    }
    if (
      rowIndex >= 3 &&
      ["harbor", "marinskyrl", "vllm-gpu", "tpu-inference"].includes(nightlyLane.id)
    ) {
      return emptyCell(nightlyLane.id, date, "not-introduced");
    }
    if (rowIndex >= 4 && nightlyLane.id === "evalchemy") {
      return emptyCell(nightlyLane.id, date, "not-introduced");
    }
    const duration = nightlyLane.expectedDuration?.minSeconds ?? 10 * MINUTES;
    return runCell(nightlyLane.id, date, 200 + rowIndex * 20 + laneIndex, duration, nightlyLane.expectedDuration ? "normal" : "baseline-pending");
  });
}

export const NIGHTLIES_FIXTURE: NightlyResponse = {
  generatedAt: NOW,
  lanes: NIGHTLY_LANES_FIXTURE,
  rows: DAILY_DATES.map(([date, label], index) => ({
    date,
    label,
    cells: index === 0 ? TODAY_CELLS : historicalCells(date, index),
  })),
  today: { healthy: 6, due: 10 },
};

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
