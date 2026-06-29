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
    deploy.py                    # `python -m ducky.deploy` — submits the Iris job (ports=["ducky"])
  tests/
    test_runner.py
    test_server.py
```

## Constants / config (`config.py`)

```python
@dataclass(frozen=True)
class DuckyConfig:
    """Resolved once at startup; no env reads after construction."""
    region: str                  # service region, e.g. "us-east5" (operational pin; not enforced in-runner)
    scratch_bucket: str          # gs:// prefix for spilled results, same region; e.g. "gs://marin-ducky-us-east5"
    # one secret set per backend, keyed by URL scheme so a single SECRET each
    # disambiguates: gs:// -> GCS, r2:// -> R2, s3:// -> CoreWeave. DuckDB httpfs
    # cannot use GCP ADC, so GCS needs HMAC interop keys.
    gcs_hmac_key_id: str         # GCS interop HMAC key id   -> CREATE SECRET ... TYPE GCS  (gs://)
    gcs_hmac_secret: str         # GCS interop HMAC secret
    r2_account_id: str           # R2 account id             -> CREATE SECRET ... TYPE R2   (r2://)
    r2_access_key: str
    r2_secret_key: str
    cw_endpoint: str             # CoreWeave S3 endpoint     -> CREATE SECRET ... TYPE S3   (s3://)
    cw_access_key: str
    cw_secret_key: str
    preview_row_cap: int = 10_000 # max rows returned inline to the browser
    memory_fraction: float = 0.8  # DuckDB memory_limit = this * host RAM; headroom for Python/Arrow/OS
    result_ttl_days: int = 7      # informational; enforced by the bucket's lifecycle rule, not by ducky
    port_name: str = "ducky"      # Iris named port; bound via ctx.get_port(port_name)

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
    """DuckDB failed to plan or execute the SQL. Wraps the DuckDB message.

    Cross-region reads are not a distinct error type in v1 — they surface here as an
    httpfs authentication failure because HMAC creds are scoped to same-region buckets.
    """
```

`QueryError` maps to HTTP 400 with `{"error": "<message>"}`. Any other exception
propagates as 500 (unexpected — let it surface).

## HTTP routes (`server.py`)

Starlette app built like `lib/iris/src/iris/cluster/worker/dashboard.py:37`, reusing
`iris.cluster.dashboard_common.html_shell` and `@requires_auth`.

| Method | Path        | Auth          | Purpose |
|--------|-------------|---------------|---------|
| GET    | `/`         | `@requires_auth` | HTML page: SQL `<textarea>`, Run button, empty result area |
| POST   | `/query`    | `@requires_auth` | Run SQL, return result JSON |
| GET    | `/health`   | `@public`     | Liveness for Iris |

**`POST /query`**
- Request body (JSON): `{"sql": "<string>"}`
- 200 response (JSON):
  ```json
  {
    "columns": ["..."],
    "rows": [[...], ...],
    "total_rows": 12345,
    "truncated": true,
    "result_path": "gs://marin-ducky-us-east5/ducky/<query_id>.parquet"
  }
  ```
  (`rows` is `QueryResult.preview_rows`.)
- 400 response (JSON): `{"error": "<message>"}` for `QueryError`.

The page renders `rows` as a table; when `truncated`, shows
"showing N of M rows — full result at `result_path` (expires in {result_ttl_days}d)".

**Entrypoint:** `python -m ducky.server`. On boot: build
`DuckyConfig.from_environment()`, construct `QueryRunner`, bind
`port = iris_ctx().get_port(config.port_name)` (the named Iris port; **not** a
hardcoded port), start uvicorn on `0.0.0.0:port`, then register
`endpoint_id = ctx.registry.register("ducky", f"http://{job_info.advertise_host}:{port}",
{"job_id": ctx.job_id.to_wire()})` (`lib/iris/src/iris/client/worker_pool.py:156-165`).
On shutdown, `ctx.registry.unregister(endpoint_id)` via Starlette `on_shutdown`.

## Persisted shape

- **Spilled result:** parquet at `{scratch_bucket}/ducky/{query_id}.parquet`, one
  file per query, written by DuckDB `COPY (…) TO … (FORMAT parquet)`. No partitioning.
- **Retention:** GCS object-lifecycle rule on `scratch_bucket`, `age = result_ttl_days`,
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
    resources=ResourceSpec(cpu=ALL_CPU, memory=ALL_MEM, device=tpu_device("v6e-8")),
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
