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

The SPA has four views: leaderboard (per-model mean score over its latest version cohort, a
colour-scaled model x task heatmap with model-comparison bars and score-over-time charts,
a suite column tree to pick which evals show, and archive controls), runs (a "by launch"
grouped view expanding each serve group to its evals, plus a flat filterable table), run
detail (metrics, version + description, live iris job/attempt status, live finelog logs, a
per-sample browser, and group siblings), and status (per-prefix ingest probes).

The per-sample browser shows how each prediction was graded (the grader method, headline metric,
score, and verbatim grader detail) and highlights the picked-versus-gold answer. Agentic (Harbor)
samples reference a step trajectory by URI; the browser lazy-loads it through the artifact endpoint
and renders the agent's turns, tool calls, observations, and reward. A sample's unbounded payloads
(the trajectory, a prediction's raw exchange) live as sibling artifact files, so paging the light
columns never materializes them.

IAP is the only access gate; there is no application auth.

## API

```
GET  /healthz               liveness
GET  /api/runs?model=&eval=&user=&status=&group=&limit=   filtered run rows
GET  /api/groups?model=&user=&limit=   runs collapsed into launches (one row per group_id) with per-eval members
GET  /api/runs/{run_id}     the full record.json for one run (404 if absent)
GET  /api/runs/{run_id}/jobs           live iris job + per-task attempt status for each role
GET  /api/runs/{run_id}/logs?role=&tail=&substring=   live finelog log lines for one role
GET  /api/runs/{run_id}/samples/tasks  tasks with exported per-sample parquets
GET  /api/runs/{run_id}/samples?task=&offset=&limit=&correct=   paged sample rows
GET  /api/runs/{run_id}/samples/artifact?uri=   one run-local sample artifact (trajectory/exchange) as text
GET  /api/runs/{run_id}/group          sibling runs sharing the run's group_id
GET  /api/matrix?include_archived=   model x task matrix (per cell, per-model version cohort) + leaderboard rows
GET  /api/history?model=&task=   every run's headline score for one cell, over time
GET  /api/meta              distinct models / evals / suites / users / statuses + archived_models + current_user
GET  /api/status            store info + per-prefix ingest probes (last probe/success/error)
POST /api/refresh           run one ingest pass now; returns the /api/status payload
POST /api/models/{model_name}/archive   set a model's archive flag ({"archived": bool})
```

The primary metric per task matches on the base metric name with lm-eval's `,<filter>`
suffix stripped: the first present of `exact_match`, `accuracy`, `acc_norm`, `acc`, `pass@1`
(falling back to the alphabetically-first non-stderr metric); its paired `<base>_stderr,<filter>` is
the reported standard error. Each matrix row reflects the model's latest version cohort -- the runs
labelled with the version of its most recent launch -- so a row never mixes evals produced against
different model states; within that cohort a cell shows the latest succeeded run's score, or -- when
no run there ever succeeded -- the latest run's failure status (still linking to that run). Archived
models (a `model_state` side table the ingestor never touches) drop out of the matrix unless
`include_archived=1`. `/api/meta` echoes the IAP caller from `X-Goog-Authenticated-User-Email` as
`current_user` and groups the eval columns into suites for the column tree.

The `jobs` and `logs` endpoints use generated Connect clients to reach the Iris controller and
finelog hub by internal IP over Direct VPC egress. GCE instance discovery requires
`roles/compute.viewer` on the runtime service account. Outside the VPC (local dev) they return a `reachable: false`
payload rather than erroring, so the dashboard shows "unreachable" and falls back to the log
tails recorded on the run.

The `samples/artifact` endpoint resolves a sample's `trajectory_uri`/`exchange_uri` through fsspec,
restricted to URIs under the run's own `results_path` -- a `..` segment or an out-of-tree URI is
refused, so the endpoint cannot fetch arbitrary object storage. It size-caps each read and, like the
logs endpoint, returns a typed `{available: false, reason}` for a missing, unreadable, or oversized
object rather than a 500. Reads are cached briefly, as sample tables are.

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

The stack uses the shared `marin-iac-key` KMS secrets provider. The operator needs
`roles/cloudkms.cryptoKeyEncrypterDecrypter` on that key; no passphrase is used.
