# grafana

Grafana over finelog, as an IAP-gated Cloud Run service. One instance serves both
clusters: it reaches `finelog-marin` and `finelog-marin-dev` on their internal IPs
over Direct VPC egress, and provisions a datasource for each. `marin` is the
federation hub — the CoreWeave clusters forward their rows to it — so its
datasource sees the whole fleet; `marin-dev` sees only itself.

Dashboards and datasources are provisioned from the files in this directory.
Grafana's SQLite is ephemeral on Cloud Run, so UI edits do not persist: change the
JSON under `dashboards/` and redeploy.

## Why Cloud Run and not an Iris job

A service that monitors X should not run on X: Grafana on Iris would serve the
dashboards you need *during* an Iris incident from the thing that is down. Cloud
Run is also the path already proven in this repo — `infra/status-page` runs the
same substrate and queries finelog's internal IP over
`--vpc-egress=private-ranges-only`.

## The bridge

Panels send SQL; the bridge runs it against finelog and returns JSON rows. finelog
gates the `Query` RPC to SELECT and enforces a server-side deadline, so the bridge
does not police the query. It exists for three things Grafana and the engine
cannot do themselves:

1. Arrow to JSON. finelog's `Query` returns Arrow IPC; Grafana's Infinity
   datasource reads JSON.
2. Caching. Grafana's query caching is Enterprise-only, so a shared
   auto-refreshing dashboard would multiply through to the finelog hub. Results
   are cached with a short TTL and concurrent misses coalesce. Panels reference
   the window through the `{{from}}` / `{{to}}` macros the bridge substitutes, so
   a relative range keeps one cache key as its edges drift between refreshes.
3. Method gating. finelog admits every RPC from the same VPC, so pointing Grafana
   at finelog's host would expose `WriteRows` and `DropTable` alongside `Query`.
   The bridge calls only `Query`.

It also flattens the EAV `labels` column into `label_<key>` fields on the way
back, since DataFusion has no JSON functions and a panel cannot group by a label
in SQL; the canary panels filter labels with `contains()` for the same reason.

```
GET /{cluster}/query?sql=&from=&to=
GET /health
```

Timestamps in the result render as epoch milliseconds, which is what Grafana
plots, so a panel selects a raw or `date_bin`-ned time column without casting it.

It runs on loopback beside Grafana; Grafana's backend datasources fetch
server-side, so nothing outside the container reaches it.

## Layout

```
src/results.py         time-macro substitution, Arrow->JSON, label flatten
src/server.py          the /query API (Starlette)
src/finelog_source.py  finelog internal-IP discovery + LogClient
src/config.py          cluster targets and bridge settings
src/cache.py           TTL cache with in-flight coalescing
provisioning/          datasources + dashboard provider
dashboards/            dashboard JSON — reviewed like code
Dockerfile             grafana:13.1.0-ubuntu + the bridge venv + the Infinity plugin
entrypoint.sh          runs both; if either dies the container dies
```

Dashboards: `fleet.json` (canary + worker health, marin only), `iris.json`
(per-task and per-worker resource usage; `iris.task` has data on both clusters),
`pipelines.json` (Zephyr throughput and shard memory).

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

```bash
./deploy.sh setup    # one-time: service account + roles/compute.viewer
./deploy.sh          # build + deploy to Cloud Run, IAP-gated
```

`min-instances` and `max-instances` are both 1: Grafana's SQLite is per-instance
and ephemeral, so more than one instance means divergent alert state and dashboard
versions, while zero means no alert rules evaluate and first paint is a cold
start.

Access is granted per user or group with `roles/iap.httpsResourceAccessor`, plus a
one-time `roles/run.invoker` for the IAP service agent (see `deploy.sh setup`). IAP
is the only gate — Grafana runs anonymous Viewer, since IAP admits everyone Google
admits and is not role-scoped.

## Adding a dashboard

Drop JSON in `dashboards/` and redeploy. Panels use the Infinity datasource with
`url: /query` and an `sql` param, plus `from`/`to` set to `${__from}`/`${__to}`.
Write the window into the SQL as `{{from}}` / `{{to}}`, and bin the time axis with
`date_bin(INTERVAL '${__interval_ms} milliseconds', ts)` so Grafana sizes the
buckets to the panel — see `dashboards/iris.json`. All dashboards use the
`${cluster}` datasource variable so one serves marin and marin-dev.
