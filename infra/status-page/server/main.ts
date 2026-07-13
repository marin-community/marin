// Hono entrypoint for the Marin status page.
//
// Serves:
//   GET /api/ferry           — GitHub Actions ferry status (60s cache, last 14 runs per tier)
//   GET /api/builds          — GitHub per-commit CI rollup on main (60s cache, last 100 commits)
//   GET /api/iris            — iris controller reachability (15s cache)
//   GET /api/control-plane/health — active env Iris + finelog health history
//   GET /api/workers         — current iris worker counts (15s cache)
//   GET /api/workers/history — 24h per-region worker count from finelog (60s cache)
//   GET /api/provisioning/history — 24h provisioning success ratio from finelog (60s cache)
//   GET /api/jobs            — iris job counts for last 24h by state (60s cache)
//   GET /api/probes          — synthetic-canary checks + provisioning from finelog (60s cache)
//   GET /api/wandb           — W&B training charts for the MoE hero run (5min cache)
//   GET /api/health          — liveness probe, no upstream calls
//   GET /*                   — static assets from web/dist (production only)
//
// During dev, static serving is effectively unused because Vite serves
// web/ directly on :5173 and proxies /api/* here.

import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { TTLCache } from "./cache.js";
import { IrisPingHistory, ServiceHealthHistory } from "./history.js";
import {
  provisioningHistory,
  workersHistory,
  type ProvisioningHistoryResponse,
  type WorkersHistoryResponse,
} from "./sources/clusterHistory.js";
import {
  FERRY_GROUPS,
  fetchTierStatus,
  type FerryGroupStatus,
  type FerryTierStatus,
} from "./sources/githubActions.js";
import { fetchBuildsOnMain, type BuildsResponse } from "./sources/githubCommits.js";
import { irisStatus, pingIris, type IrisPingResult } from "./sources/iris.js";
import { jobsSnapshot, type JobsSnapshot } from "./sources/jobs.js";
import { probesSnapshot, type ProbesSnapshot } from "./sources/probes.js";
import {
  serviceHealthResponse,
  serviceHealthSample,
  serviceHealthSnapshot,
  type ServiceHealthSnapshot,
} from "./sources/serviceHealth.js";
import { wandbSnapshot, type WandbSnapshot } from "./sources/wandb.js";
import { workerSnapshot, type WorkersSnapshot } from "./sources/workers.js";

const FERRY_RUN_LIMIT = 14;
const BUILD_HISTORY = 100;

// Cache is keyed per tier workflow file so the three datakit tiers share
// the same shield as the single-workflow ferries.
const ferryCache = new TTLCache<FerryTierStatus>(60_000);
const buildCache = new TTLCache<BuildsResponse>(60_000);
const workersCache = new TTLCache<WorkersSnapshot>(15_000);
const jobsCache = new TTLCache<JobsSnapshot>(60_000);
// Worker (60s cadence) and provisioning (15min cadence) history come from the
// canary's finelog rows; a 60s shield keeps finelog query load low without
// lagging the worker series.
const workersHistoryCache = new TTLCache<WorkersHistoryResponse>(60_000);
const provisioningHistoryCache = new TTLCache<ProvisioningHistoryResponse>(60_000);
// Probe metrics turn over slowly — health checks every ≤5min, provisioning
// every 15min — so a 60s shield is plenty and keeps finelog query load low.
const probesCache = new TTLCache<ProbesSnapshot>(60_000);
// Training metrics move at eval cadence (hours); 5min keeps us polite to
// the anonymous W&B GraphQL endpoint without visibly lagging the charts.
const wandbCache = new TTLCache<WandbSnapshot>(300_000);

// Iris controller ping sampler. We probe /health on a fixed cadence and
// keep a rolling 1h window of successful samples so /api/iris can report
// p50/p90/p99 alongside the most recent latency. Failed pings are
// recorded as the latest result (so the dot can flip red) but excluded
// from the percentile window.
const IRIS_PING_INTERVAL_MS = 2_000;
const IRIS_PING_WINDOW_MS = 60 * 60 * 1000;
const IRIS_PING_CAPACITY = Math.ceil(IRIS_PING_WINDOW_MS / IRIS_PING_INTERVAL_MS);
const irisPingHistory = new IrisPingHistory(IRIS_PING_CAPACITY);
let lastIrisPing: IrisPingResult | null = null;

// In-process sampler cadence + buffer sizing for the control-plane latency
// history (worker-count history now lives in finelog — see clusterHistory.ts).
const SAMPLE_INTERVAL_MS = 30_000;
const HISTORY_CAPACITY = Math.ceil((24 * 60 * 60 * 1000) / SAMPLE_INTERVAL_MS);
const serviceHealthHistory = new ServiceHealthHistory(HISTORY_CAPACITY);
const SERVICE_HEALTH_WINDOW_MS = 24 * 60 * 60 * 1000;
let lastServiceHealth: ServiceHealthSnapshot[] = [];

async function sampleIrisPing(): Promise<void> {
  const result = await pingIris();
  lastIrisPing = result;
  if (result.reachable && result.latencyMs !== null) {
    irisPingHistory.push({
      t: Date.parse(result.fetchedAt),
      latencyMs: result.latencyMs,
    });
  } else if (result.error) {
    console.error("iris ping sampler:", result.error);
  }
}

async function sampleServiceHealth(): Promise<void> {
  const snapshots = await serviceHealthSnapshot();
  lastServiceHealth = snapshots;
  serviceHealthHistory.push(serviceHealthSample(snapshots));
}

// Kick off immediately, then on a fixed cadence. unref() lets the process
// exit cleanly during tests without waiting on the timer.
void sampleIrisPing().catch((err) => {
  console.error("iris ping sampler error", err);
});
setInterval(() => {
  void sampleIrisPing().catch((err) => {
    console.error("iris ping sampler error", err);
  });
}, IRIS_PING_INTERVAL_MS).unref();

void sampleServiceHealth().catch((err) => {
  console.error("service health sampler error", err);
});
setInterval(() => {
  void sampleServiceHealth().catch((err) => {
    console.error("service health sampler error", err);
  });
}, SAMPLE_INTERVAL_MS).unref();

const app = new Hono();

app.get("/api/health", (c) => c.json({ status: "ok" }));

app.get("/api/ferry", async (c) => {
  const groups: FerryGroupStatus[] = await Promise.all(
    FERRY_GROUPS.map(async (group) => ({
      name: group.name,
      tiers: await Promise.all(
        group.tiers.map((tier) =>
          ferryCache.get(tier.file, () => fetchTierStatus(tier, FERRY_RUN_LIMIT)),
        ),
      ),
    })),
  );
  return c.json({ runLimit: FERRY_RUN_LIMIT, groups });
});

app.get("/api/builds", async (c) => {
  const snapshot = await buildCache.get("builds", () => fetchBuildsOnMain(BUILD_HISTORY));
  return c.json(snapshot);
});

app.get("/api/iris", (c) => {
  return c.json(irisStatus(lastIrisPing, irisPingHistory.samples(), IRIS_PING_WINDOW_MS));
});

app.get("/api/control-plane/health", (c) => {
  return c.json(
    serviceHealthResponse(
      lastServiceHealth,
      serviceHealthHistory.samples(),
      SERVICE_HEALTH_WINDOW_MS,
    ),
  );
});

app.get("/api/workers", async (c) => {
  const snapshot = await workersCache.get("workers", () => workerSnapshot());
  return c.json(snapshot);
});

app.get("/api/workers/history", async (c) => {
  const snapshot = await workersHistoryCache.get("workers-history", () => workersHistory());
  return c.json(snapshot);
});

app.get("/api/provisioning/history", async (c) => {
  const snapshot = await provisioningHistoryCache.get("provisioning-history", () =>
    provisioningHistory(),
  );
  return c.json(snapshot);
});

app.get("/api/jobs", async (c) => {
  const snapshot = await jobsCache.get("jobs", () => jobsSnapshot());
  return c.json(snapshot);
});

app.get("/api/probes", async (c) => {
  const snapshot = await probesCache.get("probes", () => probesSnapshot());
  return c.json(snapshot);
});

app.get("/api/wandb", async (c) => {
  const snapshot = await wandbCache.get("wandb", () => wandbSnapshot());
  return c.json(snapshot);
});

// Static assets. Hono serves files relative to the process cwd, so the
// Dockerfile must run the server from a directory where `web/dist` is
// resolvable. The runtime stage of the Dockerfile copies web/dist next
// to server/dist and starts node from the image's /app root.
app.use(
  "/*",
  serveStatic({
    root: "./web/dist",
    // SPA fallback: any path that doesn't map to a file falls through to
    // index.html so React Router (if added later) can handle it.
    rewriteRequestPath: (path) => (path === "/" ? "/index.html" : path),
  }),
);

const port = Number(process.env.PORT ?? "8080");
serve({ fetch: app.fetch, port }, (info) => {
  console.log(`marin-infra-dashboard listening on :${info.port}`);
});
