# marin-infra-dashboard

Internal dashboard for Marin: seven-day nightly regression status from GitHub Actions,
a GitHub Build panel showing aggregate CI status for the last 100
commits on main (per-commit check-run rollup), an Iris section
surfacing controller reachability, worker counts (current + 24h
history), active-environment Iris + finelog health, and the 24h
job-state breakdown, and a Probes section reading the synthetic-canary
metrics that `infra/probes/` writes to finelog — health checks plus the
accelerator-provisioning rollup. Deployed as Cloud Run + native IAP.

## Stack

- **Server** — Node 20 + TypeScript + [Hono](https://hono.dev). Exposes
  `/api/nightlies`, `/api/builds`, `/api/iris`, `/api/workers`,
  `/api/control-plane/health`, `/api/workers/history`,
  `/api/provisioning/history`, `/api/jobs`, `/api/probes`, `/api/wandb`,
  `/api/health`, and serves the built web UI from `web/dist`.
- **Web** — Vite + React 18 + TypeScript + Jotai + `@tanstack/react-query`
  + Tailwind.
- Single `package.json`, multi-stage Dockerfile, single service account,
  single Cloud Run service.

## Layout

```
server/
  main.ts           Hono app: routes, sampler, static serving
  cache.ts          TTL cache with in-flight coalesce
  history.ts        ring buffers for the in-process iris-ping + control-plane series
  sources/
    github.ts          shared auth header + Build repository helper
    nightlyConfig.ts   ordered lane schedules, grace, lifecycle, duration ranges
    nightlyProjection.ts UTC calendar and health projection
    githubNightlies.ts cross-repository scheduled workflow runs (REST API)
    githubCommits.ts   Build panel: per-commit CI rollup on main (GraphQL)
    iris.ts            iris controller /health caller
    serviceHealth.ts   active env Iris + finelog /health probes (+ finelog URL)
    workers.ts         iris worker counts via the ListWorkers RPC
    clusterHistory.ts  24h worker + provisioning history from finelog canary rows
    jobs.ts            iris 24h job-state breakdown via ExecuteRawQuery
    probes.ts          synthetic-canary checks + provisioning from finelog
    wandb.ts           W&B training charts via the anonymous GraphQL API
    finelogQuery.ts    finelog StatsService SQL query → Arrow IPC decode
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
      useNightlies.ts react-query hook
      useBuilds.ts
      useIris.ts
      useControlPlaneHealth.ts
      useWorkers.ts
      useWorkersHistory.ts
      useProvisioningHistory.ts
      useJobs.ts
      useProbes.ts
      useWandb.ts
    components/
      NightlyPanel.tsx
      BuildPanel.tsx  GitHub CI, last 100 runs on main
      IrisPanel.tsx   wraps reachability + WorkersPanel + ControlPlanePanel + JobsPanel
      ControlPlanePanel.tsx active env Iris + finelog latency chart
      WorkersPanel.tsx  live worker counts + side-by-side availability & provisioning history
      ProvisioningHistoryChart.tsx per-region + fleet-average provisioning success ratio
      JobsPanel.tsx
      WandbPanel.tsx  W&B training charts for the MoE hero run
      ProbesPanel.tsx synthetic-canary health checks + provisioning rollup
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
npm test           # deterministic server behavior tests
npm run build      # vite bundle + tsc server compile
```

Vite proxies `/api/*` to the Hono server, so the browser sees a
same-origin app.

### Environment variables

| Variable            | Purpose                                                               |
|---------------------|-----------------------------------------------------------------------|
| `GITHUB_TOKEN`      | Required for Build GraphQL and practical Nightlies polling across 12 workflows; raises REST limits from 60/hr to 5000/hr. |
| `CONTROLLER_URL`    | Override controller discovery. Set for local dev (see below).        |
| `PROD_IRIS_URL`     | Override prod Iris health probe URL. Falls back to `CONTROLLER_URL`. |
| `DEV_IRIS_URL`      | Override dev Iris health probe URL.                                  |
| `PROD_FINELOG_URL`  | Override prod finelog health probe URL. Falls back to `FINELOG_URL`. |
| `DEV_FINELOG_URL`   | Override dev finelog health probe URL.                               |
| `FINELOG_URL`       | Legacy override for the prod finelog health probe.                   |
| `CONTROL_PLANE_ENV` | Force control-plane health probes to `prod` or `dev`. Defaults from `CLUSTER_NAME` (`marin-dev` → `dev`, otherwise `prod`). |
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

The Control Plane panel samples only the active dashboard environment:
`prod_iris` + `prod_finelog` for prod, or `dev_iris` + `dev_finelog` for
dev. For local development, set the matching URL overrides to any
tunnels you have open; unset overrides fall back to GCE internal-IP
discovery.
Raw `/health` probes run every 30s. The chart plots rolling 5-minute
p50 and max latency from those probes, so it is less jagged than
plotting every individual round trip while still showing spikes.

A reachable controller is a hard requirement — there is no offline
mode, and panels that depend on the controller will surface an error
if it's unreachable.

## Configuration

### Nightly regression lanes

`server/sources/nightlyConfig.ts` owns the fixed 12-lane order and each lane's
repository, workflow, branch, simple UTC schedule, inclusive lifecycle dates,
independent overdue grace, and optional expected-duration range with provenance.
Ranges are reviewed configuration, never learned from dashboard history. A lane
without trustworthy bounds renders `baseline pending` instead of guessing.

`githubNightlies.ts` fetches only `event=schedule` runs; manual dispatches cannot
replace or rescue a scheduled occurrence. `nightlyProjection.ts` maps parent run
`created_at` timestamps into seven UTC calendar rows, keeps raw GitHub conclusions
separate from duration confidence, and computes `Today: healthy/due`. Grace covers
time until GitHub creates a run, not its completion. Scheduled reruns show the final
attempt and retain a `failed → passed` recovery marker.

Each run belongs to the expected occurrence whose half-open window contains its
`created_at` time. A scheduler delay longer than one full lane cadence is therefore
indistinguishable from the next occurrence; the dashboard attributes it to that next
occurrence. If multiple parent runs land in one window, the latest is shown and the
earlier run links remain available in cell details.

The UI always shows status and compact duration. Purple hatch means too short,
amber means slow, stronger amber means over 1.5× the lane maximum, and gray dots
mean baseline pending. Essential state never depends on hover.

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

### Probes panel

The Probes panel renders the synthetic-canary telemetry the
`infra/probes/` daemon writes to the finelog `infra.canary.metrics`
namespace (one flat `{metric, value, labels, collected_at}` row per
sample). Two bounded SQL queries (Apache DataFusion, finelog's read
engine — note: no JSON functions, so labels are decoded app-side) run
against the **active
environment's** finelog log-server through its `StatsService.Query`
Connect RPC — the same JSON-over-HTTP shape the controller's
`ExecuteRawQuery` uses, except the result is an Arrow IPC stream, which
`server/sources/finelogQuery.ts` decodes with `apache-arrow`. No
controller hop is involved; the finelog URL is resolved by
`activeFinelogUrl()` (same overrides + GCE discovery as the
control-plane health probe). The two queries are:

- **Health checks** — the latest `probe_up` (+ matching
  `probe_latency_ms`) per synthetic check (`controller-ping`,
  `finelog-write`, `iris-job-submit/<zone>`). Only checks that emit
  `probe_up` appear; a gauge collector surfaces here only when it
  fails.
- **Provisioning** — the latest provisioning cycle's `provision_*`
  gauges, which the probe rolls up from the controller's
  `iris.provisioning` namespace over a trailing window (default 3h).
  Rendered as a fleet summary (create success ratio = ready /
  resolved attempts, create→ready latency p50/p95, ready / stockout /
  error / preempted counts, pools placing vs. stuck) plus a per-pool
  breakdown keyed by `(resource_type, scale_group, zone)`. Per-pool
  success ratio is computed client-side; preemptions are runtime
  deaths and excluded from the create-success rate. See
  `infra/probes/src/provisioning.py` for the metric vocabulary.

The panel surfaces a friendly empty state until the canary has written
metrics (e.g. before first deploy, or if the namespace is missing).

### Training panel (W&B)

The Training panel (below Workers in the Iris section) renders the
headline charts of the public W&B report
["67B-A2B MoE on 10T tokens"](https://wandb.ai/marin-community/marin_moe/reports/67B-A2B-MoE-on-10T-tokens--VmlldzoxNzM1OTMxMQ):
train cross-entropy loss, Paloma macro loss, and MFU, all against
cumulative training tokens.

`server/sources/wandb.ts` first reads the **report's spec** (the
`view(id:)` GraphQL field) and charts whatever runs the report's runset
pins — so when a new resume run is added to the report, the panel picks
it up on the next fetch with no code change. Which metrics are charted
is fixed in the `CHARTS` constant. Run history comes from the
`sampledHistory` GraphQL field (800 samples per series, one request per
run). `marin-community` is a public entity, so everything is anonymous —
**no `WANDB_API_KEY` is needed**; if the project ever goes private the
panel will surface the GraphQL error and we'd need to plumb a key.

Readability treatment in `WandbPanel.tsx`: dense series get a debiased
EMA (`SMOOTHING_ALPHA`) drawn over a faint raw trace, sparse eval series
are plotted as-is, and the y-domain is clipped at the 98th percentile so
warmup loss spikes don't squash the curve (the same effect as the y-axis
cap the report's own panels use).

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

Each Cloud Run deployment is a single active environment. The prod
service should use `CLUSTER_NAME=marin`/`CONTROL_PLANE_ENV=prod`; a dev
service should use `CLUSTER_NAME=marin-dev`/`CONTROL_PLANE_ENV=dev`
plus the dev controller discovery settings.

## Caching

| Source          | Backend TTL | Frontend `refetchInterval` | Window              |
|-----------------|-------------|----------------------------|---------------------|
| Nightlies       | 60s raw GitHub snapshots | 60s             | 7 UTC days          |
| Build           | 60s         | 60s                        | 100 commits on main |
| Iris            | 15s         | 15s                        | current only        |
| Control plane   | in-memory   | 30s                        | 24h ring buffer     |
| Workers         | 15s         | 30s                        | current only        |
| Workers history | 60s         | 60s                        | 24h from finelog    |
| Provisioning history | 60s    | 60s                        | 24h from finelog    |
| Jobs            | 60s         | 60s                        | 24h window          |
| Probes          | 60s         | 60s                        | latest cycle        |
| W&B training    | 5min        | 5min                       | full run history (sampled) |

Backend TTL is the authoritative shield against the GitHub rate limit —
frontend polling can be tuned without affecting upstream. Concurrent
backend requests for the same key coalesce into one upstream call via
`server/cache.ts`.

The Workers panel renders two finelog-backed history charts side by
side: per-region healthy worker counts (the `worker_healthy` gauge the
canary writes every 60s) and the provisioning create-success ratio
(a fleet average plus per-region lines, derived from the per-pool
`provision_ready` / `provision_outcomes` gauges; zones roll up to
regions). Both query the trailing 24h via `server/sources/clusterHistory.ts`
and survive Cloud Run restarts since the history lives in finelog, not in
process. The remaining in-process ring buffers (`server/history.ts`) back
only the iris-ping and control-plane latency series, filled by a
background sampler on a fixed cadence so they keep ticking even when
nobody is looking at the dashboard.

## Controller data

The Workers and Jobs panels read from the controller via Connect RPC:

- **Workers** — `ListWorkers` (paged at 1000 / page), aggregated
  client-side. Worker liveness moved from SQLite to in-memory after
  PR #5559, so a raw-SQL aggregation is no longer possible.
- **Jobs** — `GROUP BY state` over `jobs WHERE submitted_at_ms > now-24h`
  via the `ExecuteRawQuery` raw-SQL RPC, with the integer enum
  translated via `server/sources/jobs.ts` (kept in sync with
  `lib/iris/src/iris/rpc/job.proto:182`).

Both RPCs nominally require the `admin` role, but the marin cluster
runs in null-auth mode
(`lib/iris/src/iris/cluster/controller/auth.py`, `NullAuthInterceptor`
promotes anonymous callers to admin), so no token is needed today.
**If auth ever gets enabled on the marin controller, both panels will
break** — we'll need to plumb a service-account bearer token.

## Known limitations

- **History depends on the canary.** Worker and provisioning history are
  read from the `infra.canary.metrics` finelog namespace the `infra/probes`
  daemon writes, so both charts are durable across Cloud Run restarts — but
  they only have data for an environment whose canary is running. Point the
  dashboard at an environment with no canary and both charts show their
  empty state rather than data.
- **Iris panel reachability row** is still `/health`-only. Worker counts
  and job-state breakdowns are surfaced in the Workers and Jobs
  subsections via `ExecuteRawQuery` SQL. Tasks, autoscaler, and detailed
  state are available via other Connect RPC methods (all support JSON
  natively) but would need additional SQL or direct RPC calls wired up.
- **Single active environment per deployment.** Prod and dev should run
  as separate Cloud Run services so each dashboard keeps its own worker,
  job, and control-plane history.
- **The Training panel's metrics are hardcoded.** Runs follow the parent
  report's runset automatically, but the charted metrics live in the
  `CHARTS` constant in `server/sources/wandb.ts` — if the report's
  headline panels change, update it by hand. The run list also assumes
  the report keeps pinning runs in its first panel grid's runset
  (`selections.tree`); a switch to filter-based selection would surface
  as a panel error.
- **Max one instance.** More than one would split the TTL cache and push
  GitHub + controller traffic up by N×. If we ever need scale, move caching
  out of process (Cloud Memorystore) or pre-compute into GCS.
