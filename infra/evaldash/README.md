# evaldash

A benchmark panel and browsable run log over every Marin eval run.

Eval runs write one canonical JSON record per run under `gs://marin-eval-metadata/evals` for GCP or
`s3://marin-us-east-02a/marin/evals` for CoreWeave. The run directory also contains the evaluator's
results and per-sample artifacts. A background loop scans the roots in `RECORDS_PREFIXES` (CW
credentials come from Secret Manager; endpoint/addressing via
`rigging.filesystem.s3_compat`) and upserts records into a Cloud SQL Postgres index
(`hai-gcp-models:us-central1:marin-metadata`, database `evals`). A Starlette app serves a JSON API over
that index and the built Vue SPA. Served at https://evaldash.oa.dev.

The default scan also includes the former flat `gs://marin-eval-metadata/runs` and
`s3://marin-us-east-02a/marin/eval-metadata/runs` roots because older CLI checkouts still write there.
Canonical `evals` roots have precedence when the same migrated `run_id` exists in both locations.

Record discovery uses a delimiter-based directory listing and checks only `*/record.json`. It does not
recursively enumerate results, samples, trajectories, or other evaluator payloads. It reads candidate
record bodies with up to 16 concurrent object-store requests. Successful records are cached by
immutable object path, so later ingest passes fetch only new records; directory listings still detect
additions and deletions.

The SPA has four views: the panel (one row per model, one column per benchmark, each cell a score
with its 95% interval and coverage badge, plus a suite column tree, run-metadata filters, a cohort
selector, an opt-in qualified aggregate, and archive controls), runs (a "by launch" grouped view
expanding each serve group to its evals, plus a flat filterable table), run detail (grade with its
interval and ungraded-item breakdown, metrics, version + description, live iris job/attempt status,
live finelog logs, a per-sample browser, and group siblings), and status (per-prefix ingest probes).

The per-sample browser shows how each prediction was graded (the grader method, headline metric,
score, and verbatim grader detail) and highlights the picked-versus-gold answer. Agentic (Harbor)
samples reference a step trajectory by URI; the browser lazy-loads it through the artifact endpoint
and renders the agent's turns, tool calls, observations, and reward. A sample's one unbounded
payload, the trajectory, lives as an archive blob referenced by URI, so paging the light columns
never materializes it.

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
GET  /api/runs/{run_id}/samples/artifact?uri=   one run-local sample artifact (the trajectory) as text
POST /api/runs/{run_id}/samples/review   LLM failure-mode review of up to n sampled task rows ({task, filter, n})
GET  /api/runs/{run_id}/group          sibling runs sharing the run's group_id
GET  /api/models/{model}    one model's aggregated detail (identity, version cohorts, current cohort cells, per-eval history, all runs; 404 if absent)
GET  /api/panel?benchmarks=&cohort=&complete=&min_coverage=&aggregate=&model=&<facet>=&include_archived=   the model x benchmark panel: per-cell measurements with intervals, explained gaps, and an optional qualified aggregate
GET  /api/compare?models=a,b[,c,d]&<panel filters>   head-to-head: per-benchmark cells, each model's difference interval against that benchmark's leader, and each model's aggregate over the shared benchmarks
GET  /api/history?model=&task=   every run's headline score for one cell, over time
GET  /api/meta              distinct models / evals / suites / users / statuses / versions + filter facets + archived_models + current_user
GET  /api/status            store info + per-prefix ingest probes (last probe/success/error)
POST /api/refresh           run one ingest pass now; returns the /api/status payload
POST /api/models/{model_name}/archive   set a model's archive flag ({"archived": bool})
```

`/api/panel` answers one selection request against the statistics engine in
`marin.evaluation.eval_stats`, which the eval runners share, so the dashboard and the producers apply
the same rules. Every cell is a measurement: the rate over the items a run graded, the item counts
behind it, the per-cause counts of items it attempted and never graded, and a 95% interval. When a run
reports an attempted count, the interval covers the ungraded items by Manski bounds with an
Imbens-Manski critical value rather than imputing them; when it reports none, the interval is labelled
`sampling_only`, because completeness is then unknown. Rankings sort on the interval's lower bound, so
losing items cannot buy rank.

Both runners establish that attempted count. Harbor reports the trials it dispatched; lm-eval publishes
no such count in its aggregate results, so the evalchemy path derives it from the document indices in
the run's own per-sample rows (`marin.evaluation.lm_eval_samples`), which also carry the pass tally and
the count of items whose grader extracted no answer. A cell every one of whose graded items yielded no
extractable answer is marked `no_answers` and starred in the panel: the zero is a real score, and it is
equally consistent with a broken grader, so the dashboard reports the observation instead of ruling on
it.

The primary metric per task matches on the base metric name with lm-eval's `,<filter>` suffix
stripped: the first present of `exact_match`, `accuracy`, `acc_norm`, `acc`, `pass@1` (falling back to
the alphabetically-first non-stderr metric).

By default each benchmark is taken from the newest run that clears the request's admission rules
(`min_coverage`, default 0.9, and a succeeded status), so a cohort that re-ran only part of a model's
benchmark set does not hide results that are still the newest available for their own benchmark;
`cohort=<version>` pins every column to one launch instead. A `(model, benchmark)` with no admitted
cell is reported in `missing` with the reason and the offending run, so an empty cell is explained.
`complete=1` keeps only models covering every selected benchmark. No cross-benchmark aggregate is
produced unless `aggregate=` names a missing-data policy (`require_complete` or `bound`), and one that
is produced carries its panel, per-benchmark metrics, and policy. An unusable query value (an unknown
aggregate policy, a `min_coverage` outside `[0, 1]`) is a 400 rather than a silent fallback to a
different question. Archived models (a `model_state` side table the ingestor never touches) drop out
unless `include_archived=1`.

`/api/compare` answers the ordering question the panel cannot: per benchmark, the model with the
highest interval lower bound leads, and every other model gets an interval for its gap to that leader,
folding in both runs' sampling error and both runs' ungraded items (asymmetrically -- the opposing
run's missing items are what can move your bound). A gap interval that spans zero means these runs
cannot resolve the ordering, which is weaker than the models being equal. Its single ranking number is
the aggregate over the shared benchmarks only, under `require_complete`.

`/api/meta` echoes the IAP caller from `X-Goog-Authenticated-User-Email` as `current_user`, groups the
eval columns into suites for the column tree, and lists the filter facets a panel request accepts.

The `jobs` and `logs` endpoints use generated Connect clients to reach the Iris controller and
finelog hub by internal IP over Direct VPC egress. GCE instance discovery requires
`roles/compute.viewer` on the runtime service account. Outside the VPC (local dev) they return a `reachable: false`
payload rather than erroring, so the dashboard shows "unreachable" and falls back to the log
tails recorded on the run.

The `samples/artifact` endpoint resolves a sample's `trajectory_uri` through fsspec,
restricted to URIs under the run's own `results_path` -- a `..` segment or an out-of-tree URI is
refused, so the endpoint cannot fetch arbitrary object storage. It size-caps each read and, like the
logs endpoint, returns a typed `{available: false, reason}` for a missing, unreadable, or oversized
object rather than a 500. Reads are cached briefly, as sample tables are.

`/api/models/{model}` aggregates in one call everything the Model view needs: the model's identity from
its newest record, one cohort entry per distinct version (newest first), each eval's score-over-time
with `-smoke` suites excluded, and every run for the model (smoke included), newest first, each with
its headline measurement.

`/api/runs/{run_id}/samples/review` samples up to `n` (default 20, capped at 40) rows of the run's
`{task}` filtered by `{filter}` (`all`/`correct`/`incorrect`), renders each into a bounded text digest,
and asks one Claude call (`EVALDASH_REVIEW_MODEL`, default `claude-haiku-4-5-20251001`, keyed by
`ANTHROPIC_API_KEY`) to bucket them into a fixed failure-mode rubric with a short narrative. Like the
logs and artifact endpoints it returns `{available: false, reason}` -- never a 500 -- when the
`anthropic` SDK is missing, the key is unset, the task has no matching samples, or the reply is not
valid JSON.

## Layout

```
src/server.py          Starlette app: JSON API + SPA serving + background ingest
src/results_db.py      private Cloud SQL schema, connection, upserts, and filtered reads
src/metrics.py         panel and comparison views over the shared statistics engine
src/discovery.py       resolve a VM internal IP from a GCE list filter
src/cluster.py         Iris and finelog generated Connect clients over Direct VPC egress
src/samples.py         typed sample API responses over fsspec + pyarrow
dashboard/             Vue 3 + TypeScript SPA (rsbuild + Tailwind 4 + Observable Plot, vue-router)
Dockerfile             node build stage + python:3.12-slim server (context = repo root)
__main__.py            Pulumi entry point — the Cloud Run service (iac.gcp.cloud_run)
Pulumi.yaml            Pulumi project, run on the shared repo venv
```

## Develop

### Local mode (no database, no cluster)

`EVALDASH_STORE=local` serves every view from a local records directory with no Cloud SQL, no
CoreWeave credentials, and no Iris/finelog access — the fastest way to iterate on the UI. Records
and their per-sample parquet are read straight from `RECORDS_PREFIXES`; the live job/log panels
degrade to "unreachable" exactly as they do off-VPC. `infra/evaldash/src/fixtures.py` writes a
deterministic sample dataset (several models across version cohorts; multiple-choice, generation,
and agentic samples; success/eval-failure/infra-failure/ungraded cases) in the on-disk layout the
reader expects.

```bash
npm --prefix infra/evaldash/dashboard install
npm --prefix infra/evaldash/dashboard run build

# Generate fixtures, then serve them.
PYTHONPATH=lib/marin/src uv run --with fsspec --with pyarrow --with pydantic \
  python infra/evaldash/src/fixtures.py /tmp/evaldash-fixtures

EVALDASH_STORE=local \
RECORDS_PREFIXES=/tmp/evaldash-fixtures \
EVALDASH_DASHBOARD_DIST=infra/evaldash/dashboard/dist \
PORT=8080 \
PYTHONPATH=infra/evaldash/src:lib/marin/src:lib/rigging/src \
uv run --with starlette --with uvicorn --with sqlalchemy --with pyarrow --with pydantic --with fsspec \
  --with anthropic \
  python infra/evaldash/src/server.py
# → http://localhost:8080  (binds loopback in local mode)
# The samples/review endpoint needs `anthropic` importable and ANTHROPIC_API_KEY set; without either it
# degrades to {available: false, reason}. EVALDASH_REVIEW_MODEL overrides the default review model.
```

### Against the shared Postgres index

```bash
# EVAL_DB_* defaults to the shared hai-gcp-models:us-central1:marin-metadata/evals instance;
# EVAL_DB_PASSWORD comes from the cloudsql-evals-password secret when unset.
RECORDS_PREFIXES=/path/to/records \
EVALDASH_DASHBOARD_DIST=infra/evaldash/dashboard/dist \
PORT=8080 \
PYTHONPATH=lib/marin/src:lib/iris/src:lib/finelog/src \
uv run \
  --with anthropic \
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

The shared Cloud Run component admits the OpenAthena Workspace domain and the Loom VM service
account through IAP on every internal site. It registers the Marin desktop OAuth client as a
programmatic audience. The stack's `viewers` list contains only additional accounts or groups
needed by evaldash.
