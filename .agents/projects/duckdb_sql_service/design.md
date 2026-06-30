# ducky — ad-hoc DuckDB SQL over a v6e host

_Why are we doing this? What's the benefit?_

We frequently need to run a one-off analytical SQL query over parquet sitting in
object storage (GCS / R2) — checking a dataset's row counts, schemas, or
distributions — and today that means hand-rolling a DuckDB script and finding a
big enough box to run it on. `ducky` is a small always-on Iris service that
exposes a dashboard: paste SQL, hit run, get results. The query executes in an
embedded DuckDB on a **full v6e-8 host** (~180 vCPU / ~1.4 TB RAM), so queries
that would OOM or crawl on a laptop finish quickly with no setup. It is
deliberately minimal — a textarea, a run button, a result table — in the spirit of
`finelog`'s `StatsService.Query` (`lib/finelog/rust/proto/finelog_stats.proto:176`),
but Python + DuckDB and standalone.

## Background

finelog is the closest prior art but is **not** a reuse target: it is now a Rust
binary using **DataFusion**, not DuckDB (`lib/finelog/rust/src/query/mod.rs`), and
deploys as a GCP VM / k8s Deployment rather than on Iris. The only live DuckDB in
the repo is finelog's deploy-CLI helper (`lib/finelog/pyproject.toml:10`) and some
ops scripts. So ducky's engine is built fresh; what we copy from finelog is the
*shape* (single `sql` string → result) and from Iris the dashboard idiom. Full
findings — including the iris resource model, endpoint proxy, and v6e topology — in
[`research.md`](./research.md).

## Challenges

- **Use the whole host.** DuckDB defaults don't grab all cores/RAM. We must read
  the host envelope at runtime — `TaskResources.from_environment()`
  (`lib/iris/src/iris/env_resources.py:58`) — and set DuckDB `threads` and
  `memory_limit` accordingly. There is no hardcoded v6e CPU/RAM constant in the
  repo; it's discovered, not declared.
- **Arbitrary SQL reads arbitrary buckets.** A `read_parquet('gs://…')` can pull
  data cross-region, which is a real cost driver per `AGENTS.md`. We need a
  same-region guardrail, not just trust.
- **Big results, run the SQL once.** A `SELECT *` can return more than a browser
  can hold. We `COPY (<sql>) TO '<parquet>'` **once**, then read the written parquet
  back for the row count and the capped preview — never execute the user's SQL
  twice, or non-deterministic queries (`random()`, `now()`, `LIMIT` without
  `ORDER BY`) would diverge between the spilled file and the preview.
- **DuckDB ↔ object-store auth.** ducky targets three stores via `httpfs`: GCS, R2,
  and CoreWeave object store. R2 and CoreWeave are native S3 (`CREATE SECRET … TYPE
  S3` with their endpoint + access/secret keys). GCS is the awkward one — DuckDB
  reads `gs://` only through the S3-compat interop API with **HMAC keys** (`TYPE GCS
  KEY_ID/SECRET`); it does *not* consume the host's GCP ADC / workload-identity
  creds. So each backend gets its own DuckDB `SECRET` from injected credentials.
- **New deployment shape.** A long-running, dashboard-exposing Iris job isn't
  templated yet; it needs a named Iris port and registry registration (not a
  hardcoded port), and reachability goes through the controller endpoint proxy.

## Costs / Risks

- **Holds a scarce v6e host 24/7 with the TPU idle.** DuckDB is CPU/RAM only — the
  8 TPU chips do nothing. This is the explicit ask (big CPU/RAM envelope), but it
  is the main cost and worth a sunset/parking story if the host is contended.
- **Arbitrary-SQL blast radius.** Whatever object storage the host's credentials
  can reach is queryable by anyone who can reach the dashboard. Auth is the proxy's
  `@requires_auth` only; there is no per-query authorization.
- **One query at a time.** A single embedded DuckDB connection; concurrent heavy
  queries contend. Acceptable for an internal ad-hoc tool, not a shared API.
- **New service to operate** (deploy, logs, restart) with no immediate user-visible
  product improvement beyond convenience.

## Design

A single Python package `lib/ducky` with three pieces:

**1. Query runner** (`runner.py`) — pure compute, no web concerns. One DuckDB
connection for the process lifetime (single query at a time — no pool). On startup
it installs/loads `httpfs`, creates one DuckDB `SECRET` per backend (GCS HMAC, R2
S3, CoreWeave S3) from injected creds, and pins `threads` + a headroom-limited
`memory_limit` (`MEMORY_FRACTION`, default 0.8
of host RAM — leaving room for Python/Arrow/OS) from
`TaskResources.from_environment()`. `run_query(sql, query_id)` runs the user SQL
**once** via `COPY (<sql>) TO 'gs://<scratch>/ducky/<query_id>.parquet' (FORMAT
parquet)`, then reads that parquet back for `total_rows` and the first
`PREVIEW_ROW_CAP` (default 10k) rows. The scratch prefix uses marin's existing
`tmp/ttl=Nd/` lifecycle convention (e.g. `gs://marin-us-east5/tmp/ttl=7d`, with N =
`RESULT_TTL_DAYS`, default 7) so spilled results auto-expire
— the service never deletes, it only writes. Returns
`QueryResult(columns, preview_rows, total_rows, truncated, result_path)`.

**2. Dashboard** (`server.py`) — a Starlette app mirroring the iris worker dashboard
(`lib/iris/src/iris/cluster/worker/dashboard.py:37`), reusing `@requires_auth`
(`lib/iris/src/iris/cluster/dashboard_common.py:39`). Queries run **asynchronously**,
because the dashboard is reached through the controller endpoint proxy, which caps
every forwarded request at 30 s (`endpoint_proxy.py:71`,
`PROXY_TIMEOUT_SECONDS = 30.0`). A synchronous `POST /query` that blocks on a
minutes-long DuckDB scan — the whole point of running on a big host — would be
killed by the proxy. So:
- `GET /` — HTML page: a CodeMirror SQL-highlighted editor, a Run button, a result
  area that shows the row count, a **cached/computed** badge, and the result's GCS
  location.
- `POST /query` — body `{sql}`; a `QueryManager` submits the SQL to a single-worker
  executor (one DuckDB query at a time) and returns `{query_id}` immediately (202).
  Identical SQL is served from an in-memory result cache (reusing the prior spilled
  parquet); otherwise the query runs in the background and spills as before.
- `GET /result/{query_id}` — returns `{status: "running"}`, `{status: "error", error}`,
  or `{status: "done", columns, rows, total_rows, truncated, result_path, cached}`.
  `result_path` is the spilled full result's GCS location; `cached` says whether it
  came from the cache. The page polls every second; each poll returns well under the
  30 s proxy window while the query itself may run for minutes.
- `GET /health` — liveness for Iris.

The endpoint registers under the cluster-global name `/ducky` (leading slash bypasses
job-namespacing), so the dashboard is reachable at a clean `/proxy/ducky/`. Query
state and the result cache live in memory for the process lifetime; ducky is
stateless and restartable, so a restart drops both.

**3. Deploy** — ducky runs as a long-running Iris job that blocks on `uvicorn`.
Because a routable service needs a *named* Iris port and `iris job run` has no
`--ports` flag, deploy goes through a small Python submit (the same path
`worker_pool.py:440` uses, `ports=["ducky"]`), not the bare CLI:

```python
# python -m ducky.deploy
client.submit(
    entrypoint="python -m ducky.server",
    resources=ResourceSpec(cpu=ALL_CPU, memory=ALL_MEM, device=tpu_device("v6e-8")),
    ports=["ducky"], region=["us-east5"], enable_extra_resources=True, no_wait=True,
)
```

The job requests the whole v6e-8 host (`ResourceSpec`,
`lib/iris/src/iris/cluster/types.py:529`). The `cpu`/`memory` request is a declared
constant that must match the real `ct6e-standard-8t` host shape — if it overshoots
the job never schedules; if it undershoots, `TaskResources.from_environment()`
reports the smaller request and silently caps DuckDB below the host. On boot the
server binds `ctx.get_port("ducky")` (not a hardcoded port) and registers
`ctx.registry.register("ducky", f"http://{job_info.advertise_host}:{port}",
metadata)` (`lib/iris/src/iris/client/worker_pool.py:156-165`), unregistering by the
returned id on shutdown. The dashboard is then reachable at `/proxy/ducky/` through
the controller reverse proxy
(`lib/iris/src/iris/cluster/controller/endpoint_proxy.py:70`). No new proto: the
browser talks plain JSON to the Starlette routes.

**Single region (us-east5).** The job is pinned to `us-east5` (`--region`,
`lib/iris/src/iris/cli/job.py:813`), the scratch bucket lives there, and the HMAC
keys are scoped to same-region buckets. Cross-region reads then simply **fail to
authenticate** rather than silently incurring egress — the guardrail is operational
(region + creds pinning), not an in-process SQL pre-parse, which would be brittle
against views/macros. v1 ships no in-runner cross-region check.

## Testing

- **Unit (runner):** a small local parquet fixture; assert `run_query` caps the
  preview at `PREVIEW_ROW_CAP`, sets `truncated`/`total_rows` correctly, writes the
  full parquet to the (faked) scratch path, and surfaces a DuckDB error as a clean
  error result rather than a 500 stack trace.
- **Resource wiring:** assert DuckDB `threads`/`memory_limit` are derived from a
  stubbed `TaskResources`, so a whole-host task actually uses the whole host.
- **Integration (iris dev cluster):** deploy the job, `POST /query` a `SELECT
  count(*) FROM read_parquet('gs://…/small.parquet')` over a known same-region
  fixture, assert the preview, the spilled `result_path`, and reachability through
  `/proxy/ducky/`.

## Open Questions

- **Credentials.** Three secret sets to inject (GCS interop HMAC, R2 S3, CoreWeave
  S3) — where do they come from and how are they passed to the task (env vars vs a
  secret mount)? For GCS specifically, which service account mints the HMAC keys,
  and are they scoped to same-region buckets so they double as the cross-region
  guardrail?
- **Host envelope.** What are the real `ct6e-standard-8t` vCPU/RAM numbers to put in
  the `ResourceSpec` request so the job both schedules and gets the whole host?
- **Scratch bucket & TTL.** Settled to `gs://marin-us-east5/tmp/ttl=7d` (existing
  lifecycle convention; N must equal `RESULT_TTL_DAYS`). Open: should the result
  path be a signed URL / clickable download, or just the `gs://` path the user
  fetches themselves?
- **Concurrency UX.** v1 is single-query-at-a-time (one DuckDB connection, a
  single-worker executor). A second `POST /query` queues behind the running one
  (FIFO). Is that enough, or do we want a visible queue depth / "busy" signal?
- **Async result retention.** Query state is in-memory and unbounded. For an
  internal tool that's fine, but should finished states expire (TTL / LRU cap), and
  should a fresh page be able to re-attach to a `query_id` from a previous session
  (it can today, until restart)?
- **Cost parking (deferred).** Always-on is the committed v1. If the v6e host proves
  too contended to idle, a follow-up could add an idle-timeout that releases the
  host with a one-click relaunch — out of scope here, flagged for reviewer pushback.

[^proxy]: The async model is forced by the controller endpoint proxy's 30 s
    request cap (`endpoint_proxy.py:71`), confirmed live: a synchronous query over a
    `gs://` parquet that ran past 30 s returned `Upstream timeout after 30s` even
    though DuckDB kept running and the spill landed. Polling sidesteps it.
