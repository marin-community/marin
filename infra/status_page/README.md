# marin-status-page

Internal dashboard for Marin: ferry workflow status from GitHub Actions
plus iris controller (orch) reachability. Deployed as Cloud Run + native
IAP, following the `infra/iris-iap-proxy/` pattern.

## Stack

- **Server** — Node 20 + TypeScript + [Hono](https://hono.dev). Exposes
  `/api/ferry`, `/api/orch`, `/api/health`, and serves the built web UI
  from `web/dist`.
- **Web** — Vite + React 18 + TypeScript + Jotai + `@tanstack/react-query`
  + Tailwind.
- Single `package.json`, multi-stage Dockerfile, single service account,
  single Cloud Run service.

## Layout

```
server/
  main.ts           Hono app: routes + static serving
  cache.ts          TTL cache with in-flight coalesce
  sources/
    githubActions.ts   GitHub REST client + workflow config
    orch.ts            iris controller /health caller
    discovery.ts       GCE label → controller internal URL
web/
  index.html
  src/
    main.tsx        React entry + providers
    App.tsx
    api.ts          typed fetch wrappers
    state.ts        Jotai atoms (auto-refresh, expand raw)
    hooks/
      useFerry.ts   react-query hook
      useOrch.ts
    components/
      FerryPanel.tsx
      OrchPanel.tsx
    style.css       Tailwind entry
Dockerfile          multi-stage build → node:20-slim runtime
deploy.sh           Cloud Run + IAP deploy
```

## Local dev

```bash
cd infra/status_page
npm install
npm run dev
```

`npm run dev` starts two processes via `concurrently`:

- Vite on `http://localhost:5173` (HMR for React/Tailwind).
- Hono on `http://localhost:8080` via `tsx watch` (auto-restart on server
  edits).

Vite proxies `/api/*` to the Hono server, so the browser sees a
same-origin app.

### Environment variables

| Variable            | Purpose                                                               |
|---------------------|-----------------------------------------------------------------------|
| `GITHUB_TOKEN`      | Lifts GH rate limit from 60/hr to 5000/hr. Optional locally.          |
| `CONTROLLER_URL`    | Override controller discovery. Set for local dev (see below).        |
| `ORCH_FIXTURE=1`    | Short-circuit orch with canned data — useful for pure UI work.        |
| `GCP_PROJECT`       | Defaults to `hai-gcp-models`.                                         |
| `CONTROLLER_ZONE`   | Defaults to `us-central1-a`.                                          |
| `CONTROLLER_LABEL`  | GCE label for controller discovery. Defaults to `iris-marin-controller`. |
| `CONTROLLER_PORT`   | Controller HTTP port. Defaults to `10000`.                            |
| `CLUSTER_NAME`      | Display name for the orch panel. Defaults to `marin`.                 |
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

Or skip the tunnel entirely and use fixtures:

```bash
ORCH_FIXTURE=1 npm run dev
```

## Configuration

Ferry workflows live in `server/sources/githubActions.ts`:

```ts
export const FERRY_WORKFLOWS = [
  { name: "Canary ferry", file: "marin-canary-ferry.yaml" },
  { name: "Datakit smoke", file: "marin-datakit-smoke.yaml" },
] as const;
```

Add more by appending to the array. `file` is the workflow filename under
`.github/workflows/`.

History window (number of runs per workflow) and the `main`-branch filter
are also constants at the top of that file.

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

| Source  | Backend TTL | Frontend `refetchInterval` |
|---------|-------------|----------------------------|
| Ferry   | 60s         | 60s                        |
| Orch    | 15s         | 15s                        |

Backend TTL is the authoritative shield against the GitHub rate limit —
frontend polling can be tuned without affecting upstream. Concurrent
backend requests for the same key coalesce into one upstream call via
`server/cache.ts`.

## Known limitations (v1)

- **Orch panel only shows reachability + latency.** The iris controller
  only exposes `/health` as a JSON endpoint
  (`lib/iris/src/iris/cluster/controller/dashboard.py:396`). Everything
  richer (workers, jobs, autoscaler state) lives behind Connect RPC,
  which would require generating TypeScript stubs. Upgrading the panel
  to surface those fields is tracked as future work.
- **Single cluster only** (`marin`). Extending to `marin-dev` means
  adding a second Cloud Run service or making the existing one
  multi-cluster.
- **No wandb panels.** Deferred from v1 — see
  `scratch/projects/marin-status-page.md` for the rendered-metrics and
  screenshot options to pick from later.
- **Max one instance.** More than one would split the TTL cache and push
  GitHub + orch traffic up by N×. If we ever need scale, move caching
  out of process (Cloud Memorystore) or pre-compute into GCS.
