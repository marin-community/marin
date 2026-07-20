# grafana

The Marin infra dashboard, as an IAP-gated Cloud Run service: Grafana plus a bridge
that fronts four sources for its Infinity datasource — finelog SQL, the live Iris
controller, the GitHub API, and the CoreWeave k8s API servers. One instance serves
both GCE clusters, reaching `finelog-marin` / `finelog-marin-dev` and each cluster's
Iris controller on their internal IPs over Direct VPC egress, and polls the public
CKS API servers of the CW clusters read-only. `marin` is the federation hub (the
CoreWeave clusters forward their rows to it), so its finelog datasource sees the
whole fleet; `marin-dev` sees only itself.

Dashboards and datasources are provisioned from the files in this directory. Grafana's
state — users, stars, preferences, alert state, and UI-created dashboards — lives in the
shared `marin-metadata` Postgres (`infra/cloudsql`), so UI edits persist across redeploys.
The provisioned dashboards under `dashboards/` are still code: change the JSON and redeploy
to update them.

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
GET /k8s/control_plane | crashloops | pending     CW control-plane state, all clusters
GET /k8s/termination_candidates | kueue | events | health
                                                    ... one response, `cluster` column
GET /k8s/alerts/{unreachable,crashloops,          alert rows: string labels + one
     webhook_ready,degraded,stuck_gpu_pods}       numeric, >=1 row per cluster
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

k8s: the bridge polls the three CoreWeave clusters' public CKS API servers with plain
httpx GETs (paginated LISTs, bounded timeouts, one 429 retry) and a single org-wide CW
read-role bearer token from `CW_READ_TOKEN` — genuine read-only kubectl, no Secrets, no
writes. Each response aggregates every cluster with a `cluster` column: watched
control-plane components (a config constant: kueue-controller-manager, iris-controller,
traefik, cert-manager) with ready/desired/restarts/waiting state, admission-webhook
ready-endpoint counts from `discovery.k8s.io` EndpointSlices, backoff pods, pending and
scheduling-gated pods, the unadmitted Kueue backlog per queue, and recent Warning
events. It also reports pods still present at least two minutes after their API
deletion deadline, classified as node cleanup, finalizer cleanup, terminal cleanup,
unbound cleanup, or invalid timestamp. Those rows include the assigned node, GPU
request, canonical Iris task-attempt id from `IRIS_TASK_ID`, priority class, and
finalizers. The pod-level scans skip provider-managed namespaces (`cw-*`, `kube-*`):
CoreWeave's per-node daemons are thousands of pods of someone else's infrastructure,
while the namespaces we operate hold about a hundred. These are current-state reads —
the bridge stores no history; trends come from the finelog-backed rows.

The `/k8s/alerts/*` routes exist for Grafana's table-alert contract: string label
columns plus exactly one numeric column, and always at least one row per cluster — an
explicit zero when healthy — so an alert rule can never enter NoData. A cluster the
bridge cannot read becomes labeled rows (its error class: auth, network, timeout, http)
rather than an empty result: `unreachable` reports 1, the count-style routes report
zero (the unreachable rule pages instead of fabricating counts), and `webhook_ready`
reports 0 ready endpoints — which also fires the webhook rule, deliberately, since
unknown admission state is the failure class it watches. A missing `CW_READ_TOKEN`
reads as an auth failure on every cluster rather than failing the boot, which would
take Grafana down with it.

The controller and finelog IPs are resolved from GCE labels and refreshed after a
connection failure. A dead controller or GitHub returns 5xx (not empty rows) and the
failure is not cached, so a panel shows an error rather than blank data; `iris/.../health`
is the exception — it returns `reachable=false` so the panel can render the outage.

## Layout

```
src/server.py          the bridge routes (Starlette): finelog SQL, Iris, GitHub, k8s
src/finelog_source.py  finelog query over its internal IP (LogClient)
src/iris_source.py     live controller RPCs: jobs, workers, health, ad-hoc query
src/github_source.py   ferry runs and CI build rollup, precomputed
src/k8s_source.py      CW k8s API reads + the per-cluster fan-out and alert rows
src/discovery.py       GCE label -> internal IP
src/config.py          cluster targets, watched components, and bridge settings
src/cache.py           TTL cache with in-flight coalescing
src/errors.py          UpstreamError -> 5xx
provisioning/          datasources (finelog, iris, github, k8s), dashboards, alerting
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
grouped by run), `k8s.json` (current CW control-plane state from the k8s source).

## Alerting

Grafana unified alerting, provisioned entirely from the files under
`provisioning/alerting/` — contact points, the notification policy tree, and the rules.
File provisioning owns that tree: UI edits to provisioned alerting resources are
rejected by Grafana and would be overwritten by the files anyway. Change the YAML and
redeploy.

Rules page only on near-certain incidents: an unreachable cluster, a
crash-looping watched component, an admission webhook with no ready endpoints, a
degraded component, a dead Iris controller, and a GPU pod that stays node-bound and
nonterminal without finalizers for five minutes after the bridge's two-minute
overdue threshold. The stuck-pod rule groups by node and links the cordon-first
recovery skill; terminal, unbound, and finalizer-held pods stay dashboard-only.
Other workload-tier signals (gated pods, Kueue backlog, workload crashloops) are
dashboard-only because they have expected benign causes. `severity=critical` routes to `ops-critical` (email ops@openathena.ai +
Slack); `severity=warning` routes to `ops-slack` (Slack only). Every rule sets
`noDataState: Alerting` and `execErrState: Alerting`, and the alert endpoints return
explicit zeros when healthy, so silence anywhere in the pipeline pages rather than
resolving.

Alert state — pending (`for`) timers, notification dedup, silences — lives in the
shared `marin-metadata` Postgres with the rest of Grafana's state (see Deploy), so it
survives redeploys. `min=max=1` keeps a single alert evaluator.

Email is optional. SMTP is plain Gmail submission (`smtp.gmail.com:587`, STARTTLS),
sending as grafana@openathena.ai with an app password from Secret Manager; the app
sends mail itself, so deliverability (SPF, spam filtering) rests on the sending
account. The deploy enables SMTP only when the `marin-grafana-smtp-credentials`
secret exists — without it the service still deploys, the email receiver fails
silently, and critical alerts reach Slack only. After changing contact points or
their credentials, send a test notification to both receivers (Alerting → Contact
points → Test) rather than trusting config presence.

## Secrets and rotation

All secrets live in Secret Manager and reach the container as env vars via the
`CloudRunService` `secrets` field; values never enter Pulumi or git.

| Env var | Secret | Feeds |
|---|---|---|
| `GITHUB_TOKEN` | `marin-status-page-github-token` | ferry/build/nightly panels |
| `GF_DATABASE_PASSWORD` | `cloudsql-grafana-password` | Grafana's Postgres state (see Deploy) |
| `CW_READ_TOKEN` | `marin-grafana-cw-read-token` | k8s source (all CW clusters) |
| `SLACK_ALERTS_WEBHOOK` | `marin-grafana-slack-webhook` | alert contact points |
| `GF_SMTP_PASSWORD` | `marin-grafana-smtp-credentials` | Grafana SMTP (email alerts, optional) |

All but the last must exist before a deploy — Cloud Run fails to start a revision
that references a missing secret. `GF_SMTP_PASSWORD` is optional: `__main__.py`
probes for the secret and only wires it (and enables SMTP) when it exists.

`CW_READ_TOKEN` is an org-wide CoreWeave API token minted with only the `read` role
(CKS binds it to the built-in `view` ClusterRole): read-only kubectl across every
cluster in the org, no Secrets, no writes. Rotation is overlap-safe: mint a second
read-role token in the CW console, `gcloud secrets versions add` it, redeploy, then
revoke the old token. The same applies to the Slack webhook and SMTP password — add a
version, redeploy, retire the old credential.

Creating the secrets:

1. CoreWeave console → API access → new token (e.g. `grafana-observer`) with only the
   `read` role, then
   `echo -n "<token>" | gcloud secrets create marin-grafana-cw-read-token --project=hai-gcp-models --data-file=-`
2. Slack → incoming webhook for `#marin-eng`, then
   `echo -n "https://hooks.slack.com/..." | gcloud secrets create marin-grafana-slack-webhook --project=hai-gcp-models --data-file=-`
3. (optional, enables email) Gmail app password for grafana@openathena.ai, then
   `echo -n "<app-password>" | gcloud secrets create marin-grafana-smtp-credentials --project=hai-gcp-models --data-file=-`
4. Send a test notification to both `ops-critical` receivers and confirm delivery.

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
uv sync --all-packages --extra deploy                     # once: iac + Pulumi providers on the venv (pulumi lives behind marin-iac[deploy])
gcloud auth configure-docker us-central1-docker.pkg.dev   # once: let buildx push to Artifact Registry

cd infra/grafana
pulumi login gs://marin-iac-state
export PULUMI_CONFIG_PASSPHRASE="$(gcloud secrets versions access latest \
  --secret=pulumi-iac-passphrase --project=hai-gcp-models)"
# The grafana.oa.dev DNS record lives in the oa.dev Cloudflare zone; the provider
# reads this token from the environment.
export CLOUDFLARE_API_TOKEN="$(gcloud secrets versions access latest \
  --secret=cloudflare-oa-dns-token --project=hai-gcp-models)"
pulumi stack select marin-grafana                         # first time: pulumi stack init marin-grafana

# Who gets in — a bare email, a *@domain wildcard, or a qualified IAM member. Editing this
# and re-running updates only the grant, never the service.
pulumi config set --path 'viewers[0]' you@example.com

pulumi preview                                            # plan; then, once it looks right:
pulumi up
```

`pulumi up` builds the Dockerfile with buildx, pushes it digest-pinned to Artifact
Registry, and rolls the service to that digest. `min` and `max` instances are both 1: one
warm instance serves this internal dashboard, min 1 keeps alert evaluation warm and first
paint off a cold start, and max 1 avoids duplicate alert notifications from parallel
evaluators.

Grafana's state is the `grafana` database on the shared `marin-metadata` Cloud SQL Postgres
(`infra/cloudsql`). `__main__.py` reads the instance connection name from a
`pulumi.StackReference` to the `marin-cloudsql` stack, mounts the Cloud SQL connector socket
under `/cloudsql`, and hands the socket directory to `entrypoint.sh` as
`DATABASE_SOCKET_DIR`, which composes `GF_DATABASE_URL` from it (Grafana's host:port
settings reject the colons in a connection name). `GF_DATABASE_PASSWORD` comes from the
`cloudsql-grafana-password` secret. Prerequisite: bring up the `marin-cloudsql` stack and
create the `grafana` SQL user + its secret version (see `infra/cloudsql/README.md`) before
`pulumi up` here, or Grafana fails to reach its database.

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
