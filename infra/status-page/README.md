# marin-infra-dashboard

Internal dashboard for Marin: ferry workflow status from GitHub Actions,
a GitHub Build panel showing aggregate CI status for the last 100
commits on main (per-commit check-run rollup), and an Iris section
surfacing controller reachability, worker counts (current + 24h
history), and the 24h job-state breakdown. Deployed as Cloud Run +
native IAP, following the `infra/iris-iap-proxy/` pattern.

## Stack

- **Server** — Node 20 + TypeScript + [Hono](https://hono.dev). Exposes
  `/api/ferry`, `/api/builds`, `/api/iris`, `/api/workers`,
  `/api/workers/history`, `/api/jobs`, `/api/health`, and serves the
  built web UI from `web/dist`.
- **Web** — Vite + React 18 + TypeScript + Jotai + `@tanstack/react-query`
  + Tailwind.
- Single `package.json`, multi-stage Dockerfile, single service account,
  single Cloud Run service.

## Layout

```
server/
  main.ts           Hono app: routes, sampler, static serving
  cache.ts          TTL cache with in-flight coalesce
  history.ts        ring buffer for worker-count history
  sources/
    github.ts          shared REPO + auth header helper
    githubActions.ts   Ferry workflow runs (REST API)
    githubCommits.ts   Build panel: per-commit CI rollup on main (GraphQL)
    iris.ts            iris controller /health caller
    workers.ts         iris worker counts via ExecuteRawQuery
    jobs.ts            iris 24h job-state breakdown via ExecuteRawQuery
    controllerQuery.ts helper for the raw-SQL Connect RPC
    discovery.ts       GCE label → controller internal URL
web/
  index.html
  src/
    main.tsx        React entry + providers
    App.tsx
    api.ts          typed fetch wrappers
    state.ts        Jotai atoms (auto-refresh)
    hooks/
      useFerry.ts   react-query hooks
      useBuilds.ts
      useIris.ts
      useWorkers.ts
      useWorkersHistory.ts
      useJobs.ts
    components/
      FerryPanel.tsx
      BuildPanel.tsx  GitHub CI, last 100 runs on main
      IrisPanel.tsx   wraps reachability + WorkersPanel + JobsPanel
      WorkersPanel.tsx
      JobsPanel.tsx
    style.css       Tailwind entry
Dockerfile          multi-stage build → node:20-slim runtime
deploy.sh           Cloud Run + IAP deploy
```

## Local dev

```bash
cd infra/status-page
npm install
npm run dev
```

`npm run dev` starts two processes via `concurrently`:

- Vite on `http://localhost:5173` (HMR for React/Tailwind).
- Hono on `http://localhost:8080` via `tsx watch` (auto-restart on server
  edits).

Before committing, run the same checks CI enforces:

```bash
npm run lint       # ESLint 9 flat config (server + web)
npm run typecheck  # tsc against tsconfig.server.json + tsconfig.web.json
npm run build      # vite bundle + tsc server compile
```

Vite proxies `/api/*` to the Hono server, so the browser sees a
same-origin app.

### Environment variables

| Variable            | Purpose                                                               |
|---------------------|-----------------------------------------------------------------------|
| `GITHUB_TOKEN`      | Required for the Build panel (GraphQL needs auth even for public repos). Also lifts Ferry's REST rate limit from 60/hr to 5000/hr. |
| `CONTROLLER_URL`    | Override controller discovery. Set for local dev (see below).        |
| `GCP_PROJECT`       | Defaults to `hai-gcp-models`.                                         |
| `CONTROLLER_ZONE`   | Defaults to `us-central1-a`.                                          |
| `CONTROLLER_LABEL`  | GCE label for controller discovery. Defaults to `iris-marin-controller`. |
| `CONTROLLER_PORT`   | Controller HTTP port. Defaults to `10000`.                            |
| `CLUSTER_NAME`      | Display name for the Iris panel. Defaults to `marin`.                 |
| `PORT`              | Hono listen port. Defaults to `8080`.                                 |

### Reaching the real controller from a laptop

The controller lives on the VPC; your laptop can't route to it directly.
Use an IAP tunnel and point `CONTROLLER_URL` at the forwarded port:

```bash
# find the controller instance name
gcloud compute instances list \
  --project=hai-gcp-models \
  --filter="labels.iris-marin-controller=true"

# tunnel port 10000 → localhost:10000
gcloud compute start-iap-tunnel <instance-name> 10000 \
  --project=hai-gcp-models \
  --zone=us-central1-a \
  --local-host-port=localhost:10000

# in another terminal:
CONTROLLER_URL=http://localhost:10000 npm run dev
```

A reachable controller is a hard requirement — there is no offline
mode, and panels that depend on the controller will surface an error
if it's unreachable.

## Configuration

### Ferry workflows

Ferry workflows live in `server/sources/githubActions.ts`:

```ts
export const FERRY_WORKFLOWS = [
  { name: "Canary ferry", file: "marin-canary-ferry.yaml" },
  { name: "Datakit smoke", file: "marin-datakit-smoke.yaml" },
] as const;
```

Add more by appending to the array. `file` is the workflow filename
under `.github/workflows/`. The `main`-branch filter is hardcoded in
`fetchWorkflowStatus`; the 30-run history window is set in
`server/main.ts`.

### Build panel

The Build panel shows per-commit aggregate CI status for the last 100
commits on main — the same green check / red X / yellow dot that
GitHub's commits view renders next to each commit title. Backed by the
GraphQL `repository.ref.target.history[*].statusCheckRollup.state`
field; see `server/sources/githubCommits.ts`.

**Requires `GITHUB_TOKEN`** — GitHub's GraphQL API needs authentication
even for public repositories. Without a token the panel renders an
error.

Success rate is computed over **finalized** commits only — excluding
pending, expected, and commits with no checks configured — so the
number reflects actual CI pass/fail ratios rather than being dragged
down by in-flight builds.

## Deploy

```bash
# one-time setup instructions (service account, secret, IAP bindings)
./deploy.sh --setup

# actual deploy
./deploy.sh
```

`deploy.sh` uses `gcloud beta run deploy --source=.`, which builds the
Dockerfile via Cloud Build, deploys to Cloud Run with native IAP
(`--iap`), Direct VPC egress (`private-ranges-only`), and pins
`min/max-instances=1` so the in-process TTL cache stays warm.

The service is a single Cloud Run service for the `marin` cluster; add a
second deploy (and a second service account if desired) for `marin-dev`
when needed.

## Caching

| Source          | Backend TTL | Frontend `refetchInterval` | Window              |
|-----------------|-------------|----------------------------|---------------------|
| Ferry           | 60s         | 60s                        | 30 workflow runs    |
| Build           | 60s         | 60s                        | 100 commits on main |
| Iris            | 15s         | 15s                        | current only        |
| Workers         | 15s         | 30s                        | current only        |
| Workers history | in-memory   | 30s                        | 24h ring buffer     |
| Jobs            | 60s         | 60s                        | 24h window          |

Backend TTL is the authoritative shield against the GitHub rate limit —
frontend polling can be tuned without affecting upstream. Concurrent
backend requests for the same key coalesce into one upstream call via
`server/cache.ts`.

The workers history is a 2880-slot ring buffer (`server/history.ts`)
filled by a background sampler on a 30s cadence — 24h worth of points.
The sampler runs on a fixed interval, not off request traffic, so
history keeps ticking even when nobody is looking at the dashboard.

## Controller data via raw SQL

The Workers and Jobs panels both read from the controller's SQLite via
the `ExecuteRawQuery` Connect RPC
(`lib/iris/src/iris/rpc/controller.proto:635`):

- **Workers** — aggregate `healthy=1 AND active=1` vs `COUNT(*)` against
  the `workers` table.
- **Jobs** — `GROUP BY state` over `jobs WHERE submitted_at_ms > now-24h`,
  with the integer enum translated via `server/sources/jobs.ts`
  (kept in sync with `lib/iris/src/iris/rpc/job.proto:182`).

The RPC nominally requires the `admin` role, but the marin cluster runs
in null-auth mode
(`lib/iris/src/iris/cluster/controller/auth.py`, `NullAuthInterceptor`
promotes anonymous callers to admin), so no token is needed today.
**If auth ever gets enabled on the marin controller, both panels will
break** — we'll need to plumb a service-account bearer token.

## Known limitations

- **Workers history is in-process.** The ring buffer is lost on Cloud
  Run restart (deploys, migrations), so the chart shows a 24h warm-up
  window after each restart. Follow-ups to consider:
  1. Persist samples to a small GCS object (rewrite on each sample).
  2. Bump retention on the controller's `worker_resource_history` table
     — currently ~45min — and aggregate from there.
  3. Add a proper `worker_count_history` table in the controller schema
     so history lives authoritatively next to the workers table.
- **Iris panel reachability row** is still `/health`-only. Worker counts
  and job-state breakdowns are surfaced in the Workers and Jobs
  subsections; tasks / autoscaler / detailed state are still behind the
  full Connect RPC surface and would need generated TypeScript stubs
  (or more `ExecuteRawQuery` SQL).
- **Single cluster only** (`marin`). Extending to `marin-dev` means
  adding a second Cloud Run service or making the existing one
  multi-cluster.
- **No wandb panels.** Deferred from v1 — see
  `scratch/projects/marin-status-page.md` for the rendered-metrics and
  screenshot options to pick from later.
- **Max one instance.** More than one would split the TTL cache and push
  GitHub + controller traffic up by N×. If we ever need scale, move caching
  out of process (Cloud Memorystore) or pre-compute into GCS.
