# grafana

The Marin infra dashboard, as an IAP-gated Cloud Run service: Grafana plus a bridge
that fronts three sources for its Infinity datasource — finelog SQL, the live Iris
controller, and the GitHub API. One instance serves both clusters, reaching
`finelog-marin` / `finelog-marin-dev` and each cluster's Iris controller on their
internal IPs over Direct VPC egress. `marin` is the federation hub (the CoreWeave
clusters forward their rows to it), so its finelog datasource sees the whole fleet;
`marin-dev` sees only itself.

Dashboards and datasources are provisioned from the files in this directory.
Grafana's SQLite is ephemeral on Cloud Run, so UI edits do not persist: change the
JSON under `dashboards/` and redeploy.

## Why Cloud Run and not an Iris job

A service that monitors X should not run on X: Grafana on Iris would serve the
dashboards you need *during* an Iris incident from the thing that is down. Cloud Run
reaches the finelog and controller internal IPs over
`--vpc-egress=private-ranges-only` without living on the cluster it watches.

## The bridge

Grafana's Infinity datasource fetches JSON over loopback from the bridge, which fronts
three upstreams and returns flat JSON rows. It runs beside Grafana; backend datasources
fetch server-side, so nothing outside the container reaches it.

```
GET /finelog/{cluster}/query?sql=&from=&to=      finelog SQL
GET /iris/{cluster}/jobs | workers | health      live controller RPCs
GET /iris/{cluster}/query?sql=                    ad-hoc SELECT (admin/null-auth)
GET /github/ferries | builds | nightlies          GitHub REST / GraphQL
GET /health                                       bridge liveness
```

finelog: a panel sends SQL and a window; the bridge substitutes the `{{from}}` / `{{to}}`
macros, runs it against finelog's `Query` RPC (SELECT-gated and deadline-bounded there),
turns the Arrow result into JSON, and caches per (cluster, SQL, window bucket) so a
relative range keeps one cache key as its edges drift. It calls only `Query`, avoiding the
`WriteRows` / `DropTable` a direct Grafana-to-finelog datasource would also expose.
Timestamps come back as epoch milliseconds, so a panel selects a raw or `date_bin`-ned
time column without casting. finelog has JSON SQL UDFs, so a panel groups by a label in SQL
— `json_get(labels,'region')`; the bridge also flattens a `labels` column into
`label_<key>` fields.

Iris: the bridge owns each query behind a fixed endpoint and returns flat rows, so the
dashboard never sends raw admin SQL. `jobs` (root jobs by state — in-flight plus 24h
terminal) and `query` use the controller's `ExecuteRawQuery`; `workers` aggregates
`ListWorkers` (worker liveness is in-memory, not SQL); `health` is the controller
`/health`. These rely on the marin controller's null-auth mode — `ExecuteRawQuery` is
admin-only — so an authed controller would break `jobs` and the ad-hoc `query`.

GitHub: `ferries`, `builds`, and `nightlies` fan out over the Actions REST and GraphQL
APIs with a server-side token (the rate-limit shield), cached, panel fields precomputed.
`nightlies` fetches each configured nightly workflow (across the marin repo and the fork
repos), classifies each (lane, day) cell server-side — health, overdue, and duration state —
and serves the result as a wide matrix: one row per day, a per-lane status code keyed by lane
id, which the state-timeline panel renders as one row per lane over the trailing week.

The controller and finelog IPs are resolved from GCE labels and refreshed after a
connection failure. A dead controller or GitHub returns 5xx (not empty rows) and the
failure is not cached, so a panel shows an error rather than blank data; `iris/.../health`
is the exception — it returns `reachable=false` so the panel can render the outage.

## Layout

```
src/server.py          the bridge routes (Starlette): finelog SQL, Iris, GitHub
src/finelog_source.py  finelog query over its internal IP (LogClient)
src/iris_source.py     live controller RPCs: jobs, workers, health, ad-hoc query
src/github_source.py   ferry runs and CI build rollup, precomputed
src/discovery.py       GCE label -> internal IP
src/config.py          cluster targets, ferry config, and bridge settings
src/cache.py           TTL cache with in-flight coalescing
src/errors.py          UpstreamError -> 5xx
provisioning/          datasources (finelog, iris, github) + dashboard provider
dashboards/            dashboard JSON — reviewed like code
Dockerfile             grafana:13.1.0-ubuntu + the bridge venv + the Infinity plugin
entrypoint.sh          runs both; if either dies the container dies
__main__.py            Pulumi entry point — the Cloud Run service (iac.gcp.cloud_run)
Pulumi.yaml            Pulumi project, run on the shared repo venv
```

Dashboards: `infra.json` (the infra overview — builds and ferries, the Iris control
plane, probes, 24h history, and the nightly regression matrix), `fleet.json` (canary +
worker health), `iris.json`
(per-task and per-worker resource usage), `pipelines.json` (Zephyr throughput and shard
memory), `training.json` (levanter training metrics from the `telltale` namespace,
grouped by run).

## Develop

```bash
uv run pytest                     # bridge unit tests
docker build -t marin-grafana .
docker run --rm -p 3000:8080 -e PORT=8080 marin-grafana
# → http://localhost:3000 (anonymous Viewer; panels need VPC access to finelog)
```

Panels only render against the real VPC: querying needs credentials that list the
finelog VMs and a network path to them. Locally you get Grafana, the provisioned
dashboards, and a bridge that 500s on query.

## Deploy

Pulumi owns the deploy: the runtime service account and its `compute.viewer` grant, the
Artifact Registry repo and image, the Cloud Run service, and the IAP wiring. The service
and its image build come from the reusable `iac.gcp.cloud_run.CloudRunService` component
(`infra/iac`); this directory is its own Pulumi project. It runs on the shared repo venv
and shares `infra/iac`'s state backend.

```bash
uv sync --all-packages                                    # once: iac + Pulumi providers on the venv
gcloud auth configure-docker us-central1-docker.pkg.dev   # once: let buildx push to Artifact Registry

cd infra/grafana
pulumi login gs://marin-iac-state
export PULUMI_CONFIG_PASSPHRASE="$(gcloud secrets versions access latest \
  --secret=pulumi-iac-passphrase --project=hai-gcp-models)"
pulumi stack select marin-grafana                         # first time: pulumi stack init marin-grafana

# Who gets in — a bare email, a *@domain wildcard, or a qualified IAM member. Editing this
# and re-running updates only the grant, never the service.
pulumi config set --path 'viewers[0]' you@example.com

pulumi preview                                            # plan; then, once it looks right:
pulumi up
```

`pulumi up` builds the Dockerfile with buildx, pushes it digest-pinned to Artifact
Registry, and rolls the service to that digest. `min` and `max` instances are both 1:
Grafana's SQLite is per-instance and ephemeral, so more than one instance means divergent
alert state and dashboard versions, while zero means no alert rules evaluate and first
paint is a cold start.

IAP is the only gate — Grafana runs anonymous Viewer. The OAuth consent screen is
project-level and shared across the project's IAP services, so nothing per-service needs
configuring beyond the `viewers` list. The service is created IAP-gated with no viewers,
i.e. reachable by nobody until the first grant.

The ferry and build panels read the GitHub API; `GITHUB_TOKEN` comes from the
`marin-status-page-github-token` Secret Manager secret, mounted by the CloudRunService
`secrets` field (the value never enters Pulumi). The name is a holdover from the retired
`marin-infra-dashboard` status page; Grafana is now its only consumer, so keep it despite
the name. Create it once if it does not exist — a classic token with no scopes or a
fine-grained PAT scoped to public-repo read is enough:

```bash
echo -n "<paste-github-token>" | gcloud secrets create marin-status-page-github-token \
  --project=hai-gcp-models --data-file=-
```

## Adding a dashboard

Drop JSON in `dashboards/` and redeploy. Panels use the Infinity datasource with
`url: /query` and an `sql` param, plus `from`/`to` set to `${__from}`/`${__to}`.
Write the window into the SQL as `{{from}}` / `{{to}}`, and bin the time axis with
`date_bin(INTERVAL '${__interval_ms} milliseconds', ts)` so Grafana sizes the
buckets to the panel — see `dashboards/iris.json`. All dashboards use the
`${cluster}` datasource variable so one serves marin and marin-dev.
