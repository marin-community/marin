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
`PREVIEW_ROW_CAP` (default 10k) rows. The scratch bucket carries an object
lifecycle rule (TTL = `RESULT_TTL_DAYS`, default 7) so spilled results auto-expire
— the service never deletes, it only writes. Returns
`QueryResult(columns, preview_rows, total_rows, truncated, result_path)`.

**2. Dashboard** (`server.py`) — a Starlette app mirroring the iris worker dashboard
(`lib/iris/src/iris/cluster/worker/dashboard.py:37`), reusing
`dashboard_common.html_shell()` and `@requires_auth`
(`lib/iris/src/iris/cluster/dashboard_common.py:166`). Routes:
- `GET /` — HTML page: a `<textarea>` for SQL, a Run button, an empty result area.
- `POST /query` — body `{sql}`; calls `run_query`, returns JSON
  `{columns, rows, total_rows, truncated, result_path}`. The page renders `rows`
  as a table and, when `truncated`, shows "showing N of M rows — full result at
  `<result_path>` (expires in 7d)".
- `GET /health` — liveness for Iris.

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
- **Scratch bucket & TTL.** Which `us-east5` bucket, and is 7 days the right default?
  Should the result path be a signed URL / clickable download, or just a `gs://`
  path the user fetches themselves?
- **Concurrency UX.** v1 is single-query-at-a-time (one DuckDB connection). Is a
  blunt "busy, try again" enough, or do we want a tiny FIFO queue with a visible
  "running" state so a long query doesn't look hung?
- **Cost parking (deferred).** Always-on is the committed v1. If the v6e host proves
  too contended to idle, a follow-up could add an idle-timeout that releases the
  host with a one-click relaunch — out of scope here, flagged for reviewer pushback.
