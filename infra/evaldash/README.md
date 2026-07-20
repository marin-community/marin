# evaldash

A leaderboard and browsable run log over every Marin eval run.

Eval runs write one canonical JSON record per run to an object-store prefix —
`gs://marin-eval-metadata/runs/<run_id>/record.json` for GCP runs,
`s3://marin-us-east-02a/marin/eval-metadata/runs/...` (CoreWeave object storage) for CW GPU
runs. A background loop scans every prefix in `RECORDS_PREFIXES` (CW credentials come from
Secret Manager; endpoint/addressing via `rigging.filesystem.s3_compat`) and upserts records
into a Cloud SQL Postgres index (`hai-gcp-models:us-central1:marin-metadata`, database
`evals`). A Starlette app serves a JSON API over that index and the built Vue SPA. Served at
https://evaldash.oa.dev.

The SPA has four views: leaderboard (per-model mean score, a colour-scaled model x task
heatmap with model-comparison bars and score-over-time charts, group-task subtasks rolled
up), runs (filterable table), run detail (metrics, live iris job/attempt status, live
finelog logs, a per-sample browser, and group siblings), and status (per-prefix ingest
probes).

IAP is the only access gate; there is no application auth.

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
suffix stripped: the first present of `exact_match`, `accuracy`, `acc_norm`, `acc`, `pass@1`
(falling back to the alphabetically-first non-stderr metric); its paired `<base>_stderr,<filter>` is
the reported standard error. A matrix cell shows the latest succeeded run's score, or -- when
no run there ever succeeded -- the latest run's failure status (still linking to that run).
`/api/meta` echoes the IAP caller from `X-Goog-Authenticated-User-Email` as `current_user`.

The `jobs` and `logs` endpoints use generated Connect clients to reach the Iris controller and
finelog hub by internal IP over Direct VPC egress. GCE instance discovery requires
`roles/compute.viewer` on the runtime service account. Outside the VPC (local dev) they return a `reachable: false`
payload rather than erroring, so the dashboard shows "unreachable" and falls back to the log
tails recorded on the run.

## Layout

```
src/server.py          Starlette app: JSON API + SPA serving + background ingest
src/results_db.py      private Cloud SQL schema, connection, upserts, and filtered reads
src/metrics.py         primary metrics, score rollups, matrix, leaderboard, and metadata
src/discovery.py       resolve a VM internal IP from a GCE list filter
src/cluster.py         Iris and finelog generated Connect clients over Direct VPC egress
src/samples.py         typed sample API responses over fsspec + pyarrow
dashboard/             Vue 3 + TypeScript SPA (rsbuild + Tailwind 4 + Observable Plot, vue-router)
Dockerfile             node build stage + python:3.12-slim server (context = repo root)
__main__.py            Pulumi entry point — the Cloud Run service (iac.gcp.cloud_run)
Pulumi.yaml            Pulumi project, run on the shared repo venv
```

## Develop

```bash
# Build the SPA (served from dashboard/dist)
npm --prefix infra/evaldash/dashboard install
npm --prefix infra/evaldash/dashboard run build

# EVAL_DB_* defaults to the shared hai-gcp-models:us-central1:marin-metadata/evals instance;
# EVAL_DB_PASSWORD comes from the cloudsql-evals-password secret when unset.
RECORDS_PREFIXES=/path/to/records \
EVALDASH_DASHBOARD_DIST=infra/evaldash/dashboard/dist \
PORT=8080 \
PYTHONPATH=lib/marin/src:lib/iris/src:lib/finelog/src \
uv run \
  --with cloud-sql-python-connector \
  --with connect-python \
  --with google-cloud-compute \
  --with google-cloud-secret-manager \
  --with pg8000 \
  --with protobuf \
  --with pyarrow \
  --with pydantic \
  --with sqlalchemy \
  --with uvicorn \
  python infra/evaldash/src/server.py
# → http://localhost:8080  (the image copies these import-light package directories too)
```

`build:check` (`vue-tsc --noEmit && rsbuild build`) is the frontend gate.

## Deploy

Deployment is handled via Pulumi (`iac.gcp.cloud_run.CloudRunService`); this directory is its
own Pulumi project (stack `marin-evaldash`), sharing `infra/pulumi`'s state backend. It depends
on the `hai-gcp-models:us-central1:marin-metadata` Cloud SQL instance and the
`cloudsql-evals-password` secret from `infra/cloudsql` — see that project's README for
provisioning them.
