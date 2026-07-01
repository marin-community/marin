# ducky — spec

Contract layer for [`design.md`](./design.md). Pins the public surface: package
layout, runner API, HTTP routes, persisted result shape, errors, deploy command.
Implementation bodies are out of scope.

## Package layout

```
lib/ducky/
  pyproject.toml                 # package "ducky", deps: duckdb>=1.0, starlette, uvicorn, iris (env_resources, dashboard_common), pyarrow
  src/ducky/
    __init__.py
    config.py                    # DuckyConfig + constants
    runner.py                    # QueryRunner, QueryResult, errors
    server.py                    # Starlette app + `python -m ducky.server` entrypoint
    deploy.py                    # submit the Iris job (ports=["ducky"])
    client.py                    # `ducky query` — POST /query + poll /result, print table/json
    cli.py                       # `ducky` command group: `ducky deploy` + `ducky query`
  tests/
    test_runner.py
    test_server.py
    test_client.py
```

The `ducky query` CLI talks to the service over a base URL (default
`http://127.0.0.1:10000/proxy/ducky`, the local controller tunnel from
`iris cluster dashboard`): `ducky query "SELECT …"` submits, polls, and prints the
preview as a table plus a stats line; `--format json` prints the raw `/result` JSON.

## Constants / config (`config.py`)

```python
@dataclass(frozen=True)
class DuckyConfig:
    """Resolved once at startup; no env reads after construction."""
    region: str                  # service region, e.g. "us-east5" (operational pin)
    allowed_buckets: tuple[str, ...] = ()  # object-store URI prefixes a query may read; empty = allow all.
                                 # In-runner same-region guardrail: a query referencing a gs://, s3://, r2://
                                 # URI outside the allowlist raises BucketNotAllowedError (-> 400) before execution.
    scratch_bucket: str          # gs:// prefix for spilled results, same region. Use the existing
                                 # marin tmp/ttl lifecycle convention, e.g. "gs://marin-us-east5/tmp/ttl=7d";
                                 # the prefix's N days MUST equal result_ttl_days.
    # GCS via HMAC interop (DuckDB httpfs can't use GCP ADC). R2 and CoreWeave are both
    # s3://, so each is a TYPE S3 secret SCOPE-d to its bucket prefix — DuckDB routes an
    # s3:// URI to the secret with the longest matching scope (different endpoints).
    gcs_hmac_key_id: str         # GCS interop HMAC key id   -> CREATE SECRET ... TYPE GCS  (gs://)
    gcs_hmac_secret: str
    r2_endpoint: str             # R2 account endpoint host  -> TYPE S3, SCOPE s3://marin-na, URL_STYLE path
    r2_access_key: str
    r2_secret_key: str
    r2_scope: str = "s3://marin-na"
    r2_url_style: str = "path"
    cw_endpoint: str             # CoreWeave endpoint host   -> TYPE S3, SCOPE s3://marin-us-east-02a, URL_STYLE vhost
    cw_access_key: str
    cw_secret_key: str
    cw_scope: str = "s3://marin-us-east-02a"
    cw_url_style: str = "vhost"  # CoreWeave rejects path-style addressing; virtual-hosted only
    preview_row_cap: int = 10_000 # max rows returned inline to the browser
    memory_fraction: float = 0.8  # DuckDB memory_limit = this * host RAM (hard cap: no OS-OOM-kill)
    spill_directory: str = "/tmp/ducky-spill"   # local disk DuckDB spills to (out-of-core) past memory_limit
    max_concurrent_queries: int = 8  # queries run in parallel, each on its own cursor
    query_timeout: int = 600      # per-query wall-clock limit; a runaway is interrupted and frees its slot
    result_ttl_days: int = 7      # informational; enforced by the bucket's lifecycle rule, not by ducky
    port_name: str = "ducky"      # Iris named port; bound via ctx.get_port(port_name)
    endpoint_name: str = "/ducky" # registry name; leading slash = cluster-global, so the
                                  # dashboard is at /proxy/ducky/ (not a per-job namespaced path)

    @classmethod
    def from_environment(cls) -> "DuckyConfig":
        """Build from process env / Iris task env. Fail fast (ValueError) if any required field is unset."""
```

`result_ttl_days` is documentation only — ducky never deletes; the scratch bucket
carries a GCS object-lifecycle rule set out of band at deploy time. The `region`
pin is operational (job pinned to `us-east5`, HMAC keys scoped to same-region
buckets); the runner does **not** parse SQL to enforce it.

## Runner API (`runner.py`)

```python
@dataclass(frozen=True)
class QueryResult:
    columns: list[str]           # column names, in select order
    preview_rows: list[list]     # up to config.preview_row_cap rows; cells are JSON-serializable scalars
    total_rows: int              # full row count of the result (>= len(preview_rows))
    truncated: bool              # True iff total_rows > len(preview_rows)
    result_path: str             # gs:// path to the full result parquet (always written, even when not truncated)
    elapsed_ms: int              # server-side execution wall time (COPY + readback)
    result_bytes: int            # on-disk size of the spilled result parquet (sum of parquet_metadata sizes)


class QueryRunner:
    """Owns one embedded DuckDB connection for the process lifetime.

    On construction: opens DuckDB, INSTALL/LOAD httpfs, creates one `SECRET` per
    backend (GCS HMAC, R2 S3, CoreWeave S3) from config creds, pins
    `threads` to the host CPU count and `memory_limit` to
    config.memory_fraction * host RAM from
    iris.env_resources.TaskResources.from_environment(). Not safe for concurrent use
    — `run_query` is serialized by the caller (single-query-at-a-time, per design).
    """

    def __init__(self, config: DuckyConfig, resources: TaskResources | None = None) -> None: ...
    # resources defaults to TaskResources.from_environment(); injectable for tests.

    def run_query(self, sql: str, query_id: str) -> QueryResult:
        """Run `sql` exactly once and return a capped preview + spilled full result.

        Executes `COPY (<sql>) TO '{config.scratch_bucket}/ducky/{query_id}.parquet'
        (FORMAT parquet)` — the user SQL runs a single time. Then reads the written
        parquet back to get `total_rows` and the first `config.preview_row_cap` rows
        for `preview_rows`; sets `truncated = total_rows > len(preview_rows)`. The
        result is never recomputed, so non-deterministic SQL (random/now/unordered
        LIMIT) stays consistent between preview and spilled file.

        `query_id` is validated as a bare uuid4 hex before path interpolation
        (rejects anything non-`[0-9a-f]{32}`) to prevent path injection.

        Raises:
            QueryError: DuckDB raised (syntax, missing table, type, auth/region — a
                cross-region read surfaces here as an httpfs auth failure). Message is
                the DuckDB error text, no stack trace.
        """
```

`query_id` is supplied by the server (a uuid4 hex); the runner does not generate
it (keeps the runner deterministic / `Math.random`-free for testing).

**Preview cell coercion:** `preview_rows` cells are JSON-serializable scalars.
DuckDB types without a native JSON form (timestamp, decimal, interval, blob, list,
struct) are coerced to their string form (Arrow→`str`) at preview build time; the
spilled parquet keeps native types.

## Errors (`runner.py`)

```python
class DuckyError(Exception):
    """Base for ducky errors surfaced to the dashboard as a clean message."""

class QueryError(DuckyError):
    """DuckDB failed to plan or execute the SQL. Wraps the DuckDB message."""

class BucketNotAllowedError(DuckyError):
    """The SQL references an object-store URI outside config.allowed_buckets. Raised before
    execution — ducky's same-region guardrail (GCS HMAC keys can't enforce region)."""
```

Both `DuckyError` subtypes map to HTTP 400 with `{"error": "<message>"}` (the
`QueryManager` catches `DuckyError`). Any other exception propagates as 500.

## HTTP routes (`server.py`)

Starlette app built like `lib/iris/src/iris/cluster/worker/dashboard.py:37`, reusing
`iris.cluster.dashboard_common` `@requires_auth`/`@public`.

Queries are **async**: the dashboard is reached through the controller endpoint
proxy, which caps every forwarded request at 30 s (`endpoint_proxy.py:71`,
`PROXY_TIMEOUT_SECONDS`). `POST /query` therefore submits and returns a `query_id`;
the client polls `GET /result/{query_id}`. Each HTTP call returns in well under 30 s
while the query runs for as long as it needs.

| Method | Path                  | Auth             | Purpose |
|--------|-----------------------|------------------|---------|
| GET    | `/`                   | `@requires_auth` | HTML page: SQL `<textarea>`, Run, result area (polls `/result`) |
| POST   | `/query`              | `@requires_auth` | Submit SQL, return `query_id` (202) |
| GET    | `/result/{query_id}`  | `@requires_auth` | Poll query status / result |
| GET    | `/health`             | `@public`        | Liveness for Iris |

**`POST /query`**
- Request body (JSON): `{"sql": "<string>"}`
- 202 response (JSON): `{"query_id": "<uuid4 hex>"}`. The `QueryManager` submits the
  SQL to a single-worker executor (one DuckDB query at a time) and returns at once.
- 400 response (JSON): `{"error": "missing 'sql'"}` when `sql` is absent/blank.

**`GET /result/{query_id}`**
- 200 `{"status": "running"}` — still executing.
- 200 `{"status": "error", "error": "<message>"}` — `QueryError` (or an unexpected
  error) raised while running.
- 200 `{"status": "done", "columns": [...], "rows": [[...]], "total_rows": N,
  "truncated": bool, "result_path": "<gs://…>", "cached": bool, "elapsed_ms": int,
  "result_bytes": int}` — `rows` is `QueryResult.preview_rows`; `result_path` is the
  spilled full result's GCS location; `cached` is true when served from the in-memory
  result cache (identical SQL); `elapsed_ms` is execution wall time and `result_bytes`
  the spilled parquet size. (On a cache hit these reflect the original run.)
- 404 `{"error": "unknown query_id"}` — no such (or expired) query.

The page (a CodeMirror SQL-highlighted editor) polls `/result/{query_id}` every
second; on `done` it renders `rows` plus a status line with the row count, a
cached/computed badge, and the `result_path` (expires in {result_ttl_days}d).

**Query manager (`server.py`)**
```python
class QueryStatus(enum.StrEnum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

class QueryManager:
    """Runs queries one at a time in a background single-worker executor and tracks
    in-memory state. `submit(sql) -> query_id` returns immediately; `get(query_id)`
    returns the current QueryState (or None). State is process-local and unbounded;
    a restart drops it (ducky is stateless/restartable). `shutdown()` stops the
    executor on app shutdown."""
    def __init__(self, runner: QueryRunner) -> None: ...
    def submit(self, sql: str) -> str: ...
    def get(self, query_id: str) -> QueryState | None: ...
    def shutdown(self) -> None: ...
```

**Entrypoint:** `python -m ducky.server`. On boot: build
`DuckyConfig.from_environment()`, construct `QueryRunner`, bind
`port = iris_ctx().get_port(config.port_name)` (the named Iris port; **not** a
hardcoded port), start uvicorn on `0.0.0.0:port`, then register
`endpoint_id = ctx.registry.register(config.endpoint_name, f"http://{job_info.advertise_host}:{port}",
{"job_id": ctx.job_id.to_wire()})` (`lib/iris/src/iris/client/worker_pool.py:156-165`). The
leading-slash `endpoint_name` (`/ducky`) registers a cluster-global endpoint, so the
proxy resolves `/proxy/ducky/` (`endpoint_proxy.py` decode: `.`→`/`, tries `/ducky`).
On shutdown (Starlette `on_shutdown`), `ctx.registry.unregister(endpoint_id)` and
`app.state.query_manager.shutdown()`.

## Persisted shape

- **Spilled result:** parquet at `{scratch_bucket}/ducky/{query_id}.parquet`, one
  file per query, written by DuckDB `COPY (…) TO … (FORMAT parquet)`. No partitioning.
- **Retention:** the marin `tmp/ttl=Nd/` lifecycle convention auto-deletes after N
  days; pick the `scratch_bucket` prefix whose N matches `result_ttl_days`. ducky
  only writes. (Equivalently, a dedicated bucket with `age = result_ttl_days`.)
  configured at deploy time (not by ducky). ducky only writes.
- No other persisted state — ducky is stateless across restarts.

## Deploy

Always-on Iris job, single region `us-east5`. A routable service needs a **named
Iris port**, which `iris job run` cannot declare (no `--ports` flag), so deploy uses
the Python submit path (`ports=["ducky"]`, as `worker_pool.py:440` does), not the
bare CLI:

```python
# python -m ducky.deploy
client.submit(
    entrypoint="python -m ducky.server",
    resources=ResourceSpec(cpu=ALL_CPU, memory=ALL_MEM, device=tpu_device("v6e-4")),  # single-VM; v6e-8 is 2 VMs
    ports=["ducky"],
    region=["us-east5"],
    enable_extra_resources=True,
    extras=["..."],          # or a --task-image carrying the ducky package
    no_wait=True,
)
```

- `ALL_CPU` / `ALL_MEM` are declared constants matching the real `ct6e-standard-8t`
  host shape (to confirm — see design Open Questions); over-requesting fails to
  schedule, under-requesting silently caps DuckDB below the host.
- The three credential sets (GCS HMAC, R2 S3, CoreWeave S3) and `scratch_bucket`
  are injected as task env vars / secret mount consumed by
  `DuckyConfig.from_environment()`.
- The `ducky` package must be importable in the task (UV extra or custom task image).

Reachable at `/proxy/ducky/` through the controller endpoint proxy
(`lib/iris/src/iris/cluster/controller/endpoint_proxy.py:70`).

## Out of scope

- Connect-RPC / proto surface — plain JSON routes only; no `.proto` for v1.
- Idle-timeout / host parking — always-on; deferred (see design Open Questions).
- Multi-region deploy — `us-east5` only.
- Query history, saved queries, auth beyond the proxy's `@requires_auth`.
- Concurrency control beyond single-query-at-a-time (no queue in v1).
- Result formats other than parquet; signed-URL downloads.
