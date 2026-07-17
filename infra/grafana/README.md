# grafana

Grafana over finelog, as an IAP-gated Cloud Run service. One instance serves both
clusters: it reaches `finelog-marin` and `finelog-marin-dev` on their internal IPs
over Direct VPC egress, and provisions a datasource for each. `marin` is the
federation hub — the CoreWeave clusters forward their rows to it — so its
datasource sees the whole fleet; `marin-dev` sees only itself.

Dashboards and datasources are code. Grafana's SQLite is ephemeral here,
so **edits made in the UI do not persist**: change the JSON under `dashboards/`
and redeploy.

## Why Cloud Run and not an Iris job

A service that monitors X must not run on X. Grafana on Iris would have served the
dashboards you need *during* an Iris incident from the thing that was broken.
Cloud Run also happens to be the path already proven in this repo:
`infra/status-page` runs the same substrate and already queries finelog's internal
IP over `--vpc-egress=private-ranges-only`.

## The bridge

Grafana cannot read finelog directly, and the reason shapes the whole design.

1. finelog's `Query` RPC returns **Arrow IPC** (base64, over Connect-as-JSON).
   Grafana's Infinity datasource reads JSON, not Arrow.
2. finelog's query engine is **DataFusion, which has no JSON functions**. The
   `labels` column that `infra/probes` writes is a JSON *string*, so labels
   **cannot be filtered or grouped in SQL** — every panel that groups by region,
   pool, or probe has to decode them row by row.

So the bridge is not a SQL proxy; it is a small metric API that owns the SQL:

```
GET /{cluster}/series?metric=&from=&to=[&group_by=][&label.k=v]
GET /health
```

Grafana never sends SQL. That is deliberate: finelog runs whatever DataFusion
accepts and enforces its 64 MiB response cap only *after* collecting the result,
so a SQL passthrough behind IAP would be an unbounded query console onto the
fleet's only telemetry store. A fixed vocabulary means there is no caller-supplied
SQL to police.

The bridge also owns the traps that would otherwise be pasted into every panel:

- **Microseconds.** finelog stores its logical `TIMESTAMP_MS` columns at
  microsecond precision, so an `arrow_cast(collected_at,'Int64')` is epoch micros.
  Grafana plots millis; raw micros land ~58,000 years in the future.
- **Caching.** Grafana's query caching is Enterprise-only, so a shared
  auto-refreshing dashboard would multiply straight through to the finelog hub.
  Results are cached with a short TTL and concurrent misses coalesce.
- **Bounds.** Every generated query carries a time predicate and a row limit, and
  windows past `GRAFANA_BRIDGE_MAX_WINDOW_HOURS` are refused before querying.

It runs on loopback beside Grafana; Grafana's backend datasources fetch
server-side, so nothing outside the container reaches it.

## Layout

```
src/series.py          SQL generation, label decode and grouping, micros->millis
src/server.py          the metric API (Starlette)
src/finelog_source.py  GCE internal-IP discovery + LogClient
src/config.py          cluster targets (code, not env) and bridge settings
src/cache.py           TTL cache with in-flight coalescing
provisioning/          datasources + dashboard provider
dashboards/            dashboard JSON — reviewed like code
Dockerfile             grafana:13.1.0-ubuntu + the bridge venv + the Infinity plugin
entrypoint.sh          runs both; if either dies the container dies
```

## Develop

```bash
uv run pytest                     # bridge unit tests
docker build -t marin-grafana .
docker run --rm -p 3000:8080 -e PORT=8080 marin-grafana
# → http://localhost:3000 (anonymous Viewer; panels need VPC access to finelog)
```

Queries need credentials that can list the finelog VMs and network access to them,
so panels only render against the real VPC — locally you get Grafana, the
provisioned dashboards, and a bridge that 500s on query.

## Deploy

```bash
./deploy.sh setup    # one-time: service account + roles/compute.viewer
./deploy.sh          # build + deploy to Cloud Run, IAP-gated
```

`min-instances` and `max-instances` are both 1 on purpose: Grafana's SQLite is
per-instance and ephemeral, so more than one instance means divergent alert state
and dashboard versions, while zero means no alert rules evaluate and first paint is
a cold start.

Access is granted per user/group with `roles/run.invoker` (see `deploy.sh setup`).
IAP is the only gate — Grafana itself runs anonymous **Viewer**, because IAP admits
everyone Google admits and is not role-scoped.

## Adding a dashboard

Drop JSON in `dashboards/` and redeploy. Panels use the Infinity datasource with
`url: /series` and `url_options.params` naming the metric, window (`${__from}` /
`${__to}`), and either `group_by` or `label.<k>` filters — see `dashboards/fleet.json`.
Metric names are whatever `infra/probes` writes; there is no discovery endpoint yet
because no dashboard uses a query variable.
Label names come from what `infra/probes` emits (`src/cluster.py`): `scope=fleet`
rollups alongside `region=` per-region rows, `probe=` on `probe_up`.
