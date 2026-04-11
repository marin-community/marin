// Hono entrypoint for the Marin status page.
//
// Serves:
//   GET /api/ferry  — GitHub Actions ferry status (60s cache)
//   GET /api/orch   — iris controller health (15s cache)
//   GET /api/health — liveness probe, no upstream calls
//   GET /*          — static assets from web/dist (production only)
//
// During dev, static serving is effectively unused because Vite serves
// web/ directly on :5173 and proxies /api/* here.

import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { TTLCache } from "./cache.js";
import { FERRY_WORKFLOWS, fetchWorkflowStatus, type FerryWorkflowStatus } from "./sources/githubActions.js";
import { clusterStatus, type OrchStatus } from "./sources/orch.js";

const ferryCache = new TTLCache<FerryWorkflowStatus>(60_000);
const orchCache = new TTLCache<OrchStatus>(15_000);

const app = new Hono();

app.get("/api/health", (c) => c.json({ status: "ok" }));

app.get("/api/ferry", async (c) => {
  const results = await Promise.all(
    FERRY_WORKFLOWS.map((wf) => ferryCache.get(wf.file, () => fetchWorkflowStatus(wf))),
  );
  return c.json({ workflows: results });
});

app.get("/api/orch", async (c) => {
  const status = await orchCache.get("marin", () => clusterStatus());
  return c.json(status);
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
  // eslint-disable-next-line no-console
  console.log(`marin-status-page listening on :${info.port}`);
});
