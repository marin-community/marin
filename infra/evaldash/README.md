# evaldash

The Marin eval-results dashboard, as an IAP-gated Cloud Run service: a leaderboard and a
browsable run log over every eval run. Eval runs write one canonical JSON record per run to
an object-store prefix — `gs://marin-eval-metadata/runs/<run_id>/record.json` for GCP runs,
`s3://marin-us-east-02a/marin/eval-metadata/runs/...` (CoreWeave object storage) for CW GPU
runs — and a CloudSQL Postgres (`hai-gcp-models:us-central1:marin-metadata`, database
`evals`) holds indexed rows. The service scans every prefix in `RECORDS_PREFIXES` on a loop
(CW credentials come from Secret Manager; endpoint/addressing via
`rigging.filesystem.s3_compat`), upserts records into Postgres, and serves a Vue SPA plus a
small JSON API. The SPA has four views: leaderboard (per-model mean score + a colour-scaled
model x task heatmap with model-comparison bars and score-over-time charts, group-task
subtasks rolled up), runs (filterable table), run detail (metrics, live iris job/attempt
status, live finelog logs, a per-sample browser, and group siblings), and status (per-prefix
ingest probes). Served at https://evaldash.oa.dev (and the run.app URL).

## Serving without the database

The record bucket is the source of truth; Postgres is an index. The server picks a backing
store at boot: Postgres when it is reachable, otherwise an in-memory store built from the
same GCS records. Either way the ingest loop keeps an in-memory snapshot current, so the
leaderboard, run detail, and filter metadata are served from GCS even when the database is
down — the header shows a `GCS cache` badge in that mode. The run list is served from the
indexed Postgres tables when the database is up.

## API

```
GET  /healthz               liveness
GET  /api/runs?model=&eval=&user=&status=&group=&limit=   filtered run rows
GET  /api/runs/{run_id}     the full record.json for one run (404 if absent)
GET  /api/runs/{run_id}/jobs           live iris job + per-task attempt status for each role
GET  /api/runs/{run_id}/logs?role=&tail=&substring=   live finelog log lines for one role
GET  /api/runs/{run_id}/samples/tasks  tasks with exported per-sample parquets
GET  /api/runs/{run_id}/samples?task=&offset=&limit=&correct=   paged sample rows
GET  /api/runs/{run_id}/group          sibling runs sharing the run's group_id
GET  /api/matrix            model x task matrix (per cell) + leaderboard rows
GET  /api/history?model=&task=   every run's headline score for one cell, over time
GET  /api/meta              distinct models / evals / users / statuses + current_user
GET  /api/status            store info + per-prefix ingest probes (last probe/success/error)
POST /api/refresh           run one ingest pass now; returns the /api/status payload
```

The primary metric per task matches on the base metric name with lm-eval's `,<filter>`
suffix stripped: the first present of `exact_match`, `acc`, `acc_norm`, `pass@1` (falling
back to the alphabetically-first non-stderr metric); its paired `<base>_stderr,<filter>` is
the reported standard error. A matrix cell shows the latest succeeded run's score, or -- when
no run there ever succeeded -- the latest run's failure status (still linking to that run).
`/api/meta` echoes the IAP caller from `X-Goog-Authenticated-User-Email` as `current_user`.

The `jobs` and `logs` endpoints reach the Iris controller and finelog hub by internal IP over
the service's Direct VPC egress (resolved from a GCE instance filter with the runtime SA's
`roles/compute.viewer`). Outside the VPC (local dev) they degrade to a `reachable: false`
payload rather than erroring, so the dashboard shows "unreachable" and falls back to the log
tails recorded on the run.

## Auth

IAP is the only gate — there is no application auth. Cloud Run is invokable solely by the
IAP service agent, and people are admitted through IAP's `httpsResourceAccessor` role
(`marin-evaldash:viewers`). The OAuth consent screen is project-level and shared across the
project's IAP services.

## Layout

```
src/server.py          Starlette app: JSON API + SPA serving + background ingest
src/metrics.py         primary-metric + stderr selection over lm-eval metric dicts
src/discovery.py       resolve a VM's internal IP from a GCE list filter
src/cluster.py         iris job status + finelog logs over Direct VPC egress (httpx)
src/samples.py         per-sample parquet reader (fsspec + pyarrow)
dashboard/             Vue 3 + TypeScript SPA (rsbuild + Tailwind 4 + Observable Plot, vue-router)
Dockerfile             node build stage + python:3.12-slim server (context = repo root)
__main__.py            Pulumi entry point — the Cloud Run service (iac.gcp.cloud_run)
Pulumi.yaml            Pulumi project, run on the shared repo venv
```

The runtime image copies `records.py` and `results_db.py` from
`lib/marin/src/marin/evaluation/`, plus this directory's `src/*.py` server modules, as flat
top-level modules, so the Docker build context is the repo root (all COPY paths are
repo-root-relative; the repo-root `.dockerignore` applies).

## Develop

Run the server against a local directory of records — no database, no GCS needed:

```bash
# Build the SPA (served from dashboard/dist)
npm --prefix infra/evaldash/dashboard install
npm --prefix infra/evaldash/dashboard run build

# Point at a local records tree laid out as <run_id>/record.json; with no EVAL_DB_* set,
# resolve_db_config() returns None and the server uses the in-memory GCS store.
RECORDS_PREFIXES=/path/to/records \
EVALDASH_DASHBOARD_DIST=infra/evaldash/dashboard/dist \
PORT=8080 \
PYTHONPATH=lib/marin/src \
.venv/bin/python infra/evaldash/src/server.py
# → http://localhost:8080  (server.py imports marin.evaluation.{records,results_db};
#   the image copies just those modules with their package skeleton)
```

`build:check` (`vue-tsc --noEmit && rsbuild build`) is the frontend gate.

## Deploy

Pulumi owns the deploy: the runtime service account and its `cloudsql.client` /
`storage.objectViewer` grants, the Artifact Registry repo and image, the Cloud Run service,
and the IAP wiring. The service and its image build come from the reusable
`iac.gcp.cloud_run.CloudRunService` component (`infra/iac`); this directory is its own Pulumi
project on the shared repo venv, sharing `infra/iac`'s state backend.

```bash
uv sync --all-packages --extra deploy                     # once: iac + Pulumi providers (pulumi lives behind marin-iac[deploy])
gcloud auth configure-docker us-central1-docker.pkg.dev   # once: let buildx push to Artifact Registry

cd infra/evaldash
pulumi login gs://marin-iac-state
export PULUMI_CONFIG_PASSPHRASE="$(gcloud secrets versions access latest \
  --secret=pulumi-iac-passphrase --project=hai-gcp-models)"
pulumi stack select marin-evaldash                        # first time: pulumi stack init marin-evaldash

# Who gets in — a bare email, a *@domain wildcard, or a qualified IAM member.
pulumi config set --path 'viewers[0]' you@example.com

pulumi preview                                            # plan; then, once it looks right:
pulumi up
```

`pulumi up` builds the Dockerfile with buildx, pushes it digest-pinned to Artifact Registry,
and rolls the service to that digest. `min` and `max` instances are both 1: the ingest loop
runs between requests (so CPU is always allocated), and a single instance keeps one ingest
cadence.

### Prerequisites the deploy assumes

- `cloudsql-evals-password` Secret Manager secret holds the `evals` DB user's password
  (mounted as `EVAL_DB_PASSWORD`). Create it once:
  ```bash
  echo -n "<db-password>" | gcloud secrets create cloudsql-evals-password \
    --project=hai-gcp-models --data-file=-
  ```
- The CloudSQL instance `hai-gcp-models:us-central1:marin-metadata` exists with database
  `evals` and user `evals` (owned by `infra/cloudsql`).
- The `gs://marin-eval-metadata` bucket exists and the runtime SA can read it.
- `CloudRunServiceArgs.cloudsql_instances` is available in `infra/iac` — it attaches the
  CloudSQL instance to the service so the connector can dial it.
