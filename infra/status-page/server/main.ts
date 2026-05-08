// Hono entrypoint for the Marin status page.
//
// Serves:
//   GET /api/ferry           — GitHub Actions ferry status (60s cache, last 30 runs)
//   GET /api/builds          — GitHub per-commit CI rollup on main (60s cache, last 100 commits)
//   GET /api/iris            — iris controller reachability (15s cache)
//   GET /api/workers         — current iris worker counts (15s cache)
//   GET /api/workers/history — in-memory 24h worker count ring buffer
//   GET /api/jobs            — iris job counts for last 24h by state (60s cache)
//   GET /api/health          — liveness probe, no upstream calls
//   GET /*                   — static assets from web/dist (production only)
//
// During dev, static serving is effectively unused because Vite serves
// web/ directly on :5173 and proxies /api/* here.

import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { TTLCache } from "./cache.js";
import { WorkerHistory } from "./history.js";
import {
  FERRY_WORKFLOWS,
  fetchWorkflowStatus,
  type FerryWorkflowStatus,
} from "./sources/githubActions.js";
import { fetchBuildsOnMain, type BuildsResponse } from "./sources/githubCommits.js";
import { irisStatus, type IrisStatus } from "./sources/iris.js";
import { jobsSnapshot, type JobsSnapshot } from "./sources/jobs.js";
import { workerSnapshot, type WorkersSnapshot } from "./sources/workers.js";

const FERRY_HISTORY = 30;
const BUILD_HISTORY = 100;

const ferryCache = new TTLCache<FerryWorkflowStatus>(60_000);
const buildCache = new TTLCache<BuildsResponse>(60_000);
const irisCache = new TTLCache<IrisStatus>(15_000);
const workersCache = new TTLCache<WorkersSnapshot>(15_000);
const jobsCache = new TTLCache<JobsSnapshot>(60_000);

// Ring buffer for worker-count history. Sized so the buffer holds 24h of
// samples at the configured cadence. The sampler runs on a fixed interval
// below — not lazily off request traffic — so history keeps ticking even
// when nobody's watching the dashboard.
const SAMPLE_INTERVAL_MS = 30_000;
const HISTORY_CAPACITY = Math.ceil((24 * 60 * 60 * 1000) / SAMPLE_INTERVAL_MS);
const workerHistory = new WorkerHistory(HISTORY_CAPACITY);

async function sampleWorkers(): Promise<void> {
  const snapshot = await workersCache.get("workers", () => workerSnapshot());
  if (snapshot.error) {
    console.error("worker sampler: snapshot error:", snapshot.error);
    // Don't pollute history with zeros when the controller is unreachable.
    return;
  }
  const regions: Record<string, number> = {};
  for (const r of snapshot.byRegion) {
    regions[r.region] = r.healthy;
  }
  workerHistory.push({
    t: Date.parse(snapshot.fetchedAt),
    regions,
  });
}

// Kick off immediately, then on a fixed cadence. unref() lets the process
// exit cleanly during tests without waiting on the timer.
void sampleWorkers().catch((err) => {
  console.error("worker sampler error", err);
});
setInterval(() => {
  void sampleWorkers().catch((err) => {
    console.error("worker sampler error", err);
  });
}, SAMPLE_INTERVAL_MS).unref();

const app = new Hono();

app.get("/api/health", (c) => c.json({ status: "ok" }));

app.get("/api/ferry", async (c) => {
  const results = await Promise.all(
    FERRY_WORKFLOWS.map((wf) =>
      ferryCache.get(wf.file, () => fetchWorkflowStatus(wf, FERRY_HISTORY)),
    ),
  );
  return c.json({ workflows: results });
});

app.get("/api/builds", async (c) => {
  const snapshot = await buildCache.get("builds", () => fetchBuildsOnMain(BUILD_HISTORY));
  return c.json(snapshot);
});

app.get("/api/iris", async (c) => {
  const status = await irisCache.get("marin", () => irisStatus());
  return c.json(status);
});

app.get("/api/workers", async (c) => {
  const snapshot = await workersCache.get("workers", () => workerSnapshot());
  return c.json(snapshot);
});

app.get("/api/workers/history", (c) => {
  return c.json({ samples: workerHistory.samples() });
});

app.get("/api/jobs", async (c) => {
  const snapshot = await jobsCache.get("jobs", () => jobsSnapshot());
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
